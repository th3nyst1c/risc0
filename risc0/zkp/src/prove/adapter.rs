// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Mutex;

use rand::thread_rng;
use rayon::prelude::*;
use risc0_core::field::{Elem, Field};

use crate::{
    adapter::{CircuitProveDef, CircuitStepContext, CircuitStepHandler, REGISTER_GROUP_AUX},
    hal::cpu::CpuBuffer,
    prove::{
        aux::{Aux, Handler},
        executor::Executor,
        write_iop::WriteIOP,
    },
    taps::TapSet,
    ZK_CYCLES,
};

pub struct ProveAdapter<'a, F, C, S>
where
    F: Field,
    C: 'static + CircuitProveDef<F>,
    S: CircuitStepHandler<F::Elem>,
{
    exec: &'a mut Executor<F, C, S>,
    mix: CpuBuffer<F::Elem>,
    aux: CpuBuffer<F::Elem>,
    steps: usize,
}

impl<'a, F, C, CS> ProveAdapter<'a, F, C, CS>
where
    F: Field,
    C: 'static + CircuitProveDef<F>,
    CS: CircuitStepHandler<F::Elem>,
{
    pub fn new(exec: &'a mut Executor<F, C, CS>) -> Self {
        let steps = exec.steps;
        ProveAdapter {
            exec,
            mix: CpuBuffer::from(Vec::new()),
            aux: CpuBuffer::from(Vec::new()),
            steps,
        }
    }

    pub fn get_taps(&self) -> &'static TapSet<'static> {
        self.exec.circuit.get_taps()
    }

    /// Perform initial 'execution' setting control + data.
    /// Additionally, write any 'results' as needed.
    pub fn execute(&mut self, iop: &mut WriteIOP<F>) {
        iop.write_field_elem_slice(&self.exec.io.as_slice());
        iop.write_u32_slice(&[self.exec.po2 as u32]);
    }

    fn compute_aux(&mut self) {
        let args = &[
            self.exec.control.as_slice_sync(),
            self.exec.io.as_slice_sync(),
            self.exec.data.as_slice_sync(),
            self.mix.as_slice_sync(),
            self.aux.as_slice_sync(),
        ];
        let aux: Mutex<Aux<F::ExtElem>> = Mutex::new(Aux::new(self.steps));
        tracing::info_span!("step_compute_aux").in_scope(|| {
            // TODO: Add an way to be able to run this on cuda, metal, etc.
            let c = &self.exec.circuit;
            (0..self.steps - ZK_CYCLES).into_par_iter().for_each_init(
                || Handler::<F>::new(&aux),
                |aux_handler, cycle| {
                    c.step_compute_aux(
                        &CircuitStepContext {
                            size: self.steps,
                            cycle,
                        },
                        aux_handler,
                        args,
                    )
                    .unwrap();
                },
            );
        });
        tracing::info_span!("calc_prefix_products").in_scope(|| {
            aux.lock().unwrap().calc_prefix_products();
        });
        tracing::info_span!("step_verify_aux").in_scope(|| {
            let c = &self.exec.circuit;
            (0..self.steps - ZK_CYCLES).into_par_iter().for_each_init(
                || Handler::<F>::new(&aux),
                |aux_handler, cycle| {
                    c.step_verify_aux(
                        &CircuitStepContext {
                            size: self.steps,
                            cycle,
                        },
                        aux_handler,
                        args,
                    )
                    .unwrap();
                },
            );
        });
    }

    /// Perform accumulations for `Auxiliary` stage, using the iop for any RNG state.
    #[tracing::instrument(skip_all)]
    pub fn accumulate(&mut self, iop: &mut WriteIOP<F>) {
        // Make the mixing values
        self.mix = CpuBuffer::from_fn(C::MIX_SIZE, |_| iop.random_elem());
        // Make and compute aux data
        let aux_size = self
            .exec
            .circuit
            .get_taps()
            .group_size(REGISTER_GROUP_AUX);
        self.aux = CpuBuffer::from_fn(self.steps * aux_size, |_| F::Elem::INVALID);

        self.compute_aux();

        // Zero out 'invalid' entries in aux and io
        let mut aux = self.aux.as_slice_mut();
        let mut io = self.exec.io.as_slice_mut();
        for value in aux.iter_mut().chain(io.iter_mut()) {
            *value = value.valid_or_zero();
        }
        // Add random noise to end of aux and change invalid element to zero
        let mut rng = thread_rng();
        for i in self.steps - ZK_CYCLES..self.steps {
            for j in 0..aux_size {
                aux[j * self.steps + i] = F::Elem::random(&mut rng);
            }
        }
    }

    pub fn po2(&self) -> u32 {
        self.exec.po2 as u32
    }

    pub fn get_control(&self) -> &CpuBuffer<F::Elem> {
        &self.exec.control
    }

    pub fn get_data(&self) -> &CpuBuffer<F::Elem> {
        &self.exec.data
    }

    pub fn get_aux(&self) -> &CpuBuffer<F::Elem> {
        &self.aux
    }

    pub fn get_mix(&self) -> &CpuBuffer<F::Elem> {
        &self.mix
    }

    pub fn get_io(&self) -> &CpuBuffer<F::Elem> {
        &self.exec.io
    }

    pub fn get_steps(&self) -> usize {
        self.steps
    }
}
