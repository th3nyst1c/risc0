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

#![cfg_attr(not(feature = "std"), no_std)]

pub mod control_id;
mod info;
pub mod layout;
mod poly_ext;
#[cfg(feature = "prove")]
pub mod prove;
mod taps;

use hex::FromHex;
use risc0_circuit_rv32im::control_id::POSEIDON_CONTROL_ID;
use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};
use risc0_zkp::{
    adapter::{CircuitCoreDef, TapsProvider},
    core::digest::Digest,
    field::baby_bear::BabyBear,
    taps::TapSet,
};

use crate::control_id::RECURSION_CONTROL_IDS;

pub const REGISTER_GROUP_ACCUM: usize = 0;
pub const REGISTER_GROUP_CODE: usize = 1;
pub const REGISTER_GROUP_DATA: usize = 2;

pub const GLOBAL_MIX: usize = 0;
pub const GLOBAL_OUT: usize = 1;

pub const CIRCUIT: CircuitImpl = CircuitImpl::new();

/// This struct implements traits that are defined by code generated by the
/// circuit definition.
pub struct CircuitImpl;

impl CircuitImpl {
    pub const fn new() -> Self {
        Self
    }

    pub fn code_size(&self) -> usize {
        self.get_taps().group_size(REGISTER_GROUP_CODE)
    }
}

impl TapsProvider for CircuitImpl {
    fn get_taps(&self) -> &'static TapSet<'static> {
        self::taps::TAPSET
    }
}

impl CircuitCoreDef<BabyBear> for CircuitImpl {}

// Values for micro inst "opcode"
pub mod micro_op {
    pub const CONST: u32 = 0;
    pub const ADD: u32 = 1;
    pub const SUB: u32 = 2;
    pub const MUL: u32 = 3;
    pub const INV: u32 = 4;
    pub const EQ: u32 = 5;
    pub const READ_IOP_HEADER: u32 = 6;
    pub const READ_IOP_BODY: u32 = 7;
    pub const MIX_RNG: u32 = 8;
    pub const SELECT: u32 = 9;
    pub const EXTRACT: u32 = 10;
}

// Externs used by recursion circuit with native bdata types.
pub trait Externs {
    fn wom_write(&mut self, _addr: BabyBearElem, _val: BabyBearExtElem) {
        unimplemented!()
    }

    fn wom_read(&self, _addr: BabyBearElem) -> BabyBearExtElem {
        unimplemented!()
    }

    fn read_iop_header(&mut self, _count: BabyBearElem, _k_and_flip_flag: BabyBearElem) {
        unimplemented!()
    }

    fn read_iop_body(&mut self, _do_mont: BabyBearElem) -> BabyBearExtElem {
        unimplemented!()
    }
}

/// This function gets valid control IDs from the Poseidon and recursion
/// circuits
pub fn valid_control_ids() -> Vec<Digest> {
    let mut all_ids = Vec::new();
    for digest_str in POSEIDON_CONTROL_ID {
        all_ids.push(Digest::from_hex(digest_str).unwrap());
    }
    for (_, digest_str) in RECURSION_CONTROL_IDS {
        all_ids.push(Digest::from_hex(digest_str).unwrap());
    }
    all_ids
}

#[cfg(feature = "test")]
pub mod testutil {
    use rand::{thread_rng, Rng};
    use risc0_zkp::{
        adapter::{CircuitInfo, TapsProvider},
        field::{
            baby_bear::{BabyBearElem, BabyBearExtElem},
            Elem, ExtElem,
        },
        hal::{Buffer, CircuitHal, Hal},
        INV_RATE,
    };

    use crate::{CircuitImpl, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA};

    pub struct EvalCheckParams {
        pub po2: usize,
        pub steps: usize,
        pub domain: usize,
        pub code: Vec<BabyBearElem>,
        pub data: Vec<BabyBearElem>,
        pub accum: Vec<BabyBearElem>,
        pub mix: Vec<BabyBearElem>,
        pub out: Vec<BabyBearElem>,
        pub poly_mix: BabyBearExtElem,
    }

    impl EvalCheckParams {
        pub fn new(po2: usize) -> Self {
            let mut rng = thread_rng();
            let steps = 1 << po2;
            let domain = steps * INV_RATE;
            let circuit = CircuitImpl::new();
            let taps = circuit.get_taps();
            let code_size = taps.group_size(REGISTER_GROUP_CODE);
            let data_size = taps.group_size(REGISTER_GROUP_DATA);
            let accum_size = taps.group_size(REGISTER_GROUP_ACCUM);
            let code = random_fps(&mut rng, code_size * domain);
            let data = random_fps(&mut rng, data_size * domain);
            let accum = random_fps(&mut rng, accum_size * domain);
            let mix = random_fps(&mut rng, CircuitImpl::MIX_SIZE);
            let out = random_fps(&mut rng, CircuitImpl::OUTPUT_SIZE);
            let poly_mix = BabyBearExtElem::random(&mut rng);
            tracing::debug!("code: {} bytes", code.len() * 4);
            tracing::debug!("data: {} bytes", data.len() * 4);
            tracing::debug!("accum: {} bytes", accum.len() * 4);
            tracing::debug!("mix: {} bytes", mix.len() * 4);
            tracing::debug!("out: {} bytes", out.len() * 4);
            Self {
                po2,
                steps,
                domain,
                code,
                data,
                accum,
                mix,
                out,
                poly_mix,
            }
        }
    }

    fn random_fps<E: Elem>(rng: &mut impl Rng, size: usize) -> Vec<E> {
        let mut ret = Vec::new();
        for _ in 0..size {
            ret.push(E::random(rng));
        }
        ret
    }

    #[allow(unused)]
    pub(crate) fn eval_check<H1, H2, C1, C2>(hal1: &H1, eval1: C1, hal2: &H2, eval2: C2, po2: usize)
    where
        H1: Hal<Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        H2: Hal<Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        C1: CircuitHal<H1>,
        C2: CircuitHal<H2>,
    {
        let params = EvalCheckParams::new(po2);
        let check1 = eval_check_impl(&params, hal1, &eval1);
        let check2 = eval_check_impl(&params, hal2, &eval2);
        assert_eq!(check1, check2);
    }

    pub fn eval_check_impl<H, C>(params: &EvalCheckParams, hal: &H, eval: &C) -> Vec<H::Elem>
    where
        H: Hal<Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        C: CircuitHal<H>,
    {
        let check = hal.alloc_elem("check", BabyBearExtElem::EXT_SIZE * params.domain);
        let code = hal.copy_from_elem("code", &params.code);
        let data = hal.copy_from_elem("data", &params.data);
        let accum = hal.copy_from_elem("accum", &params.accum);
        let mix = hal.copy_from_elem("mix", &params.mix);
        let out = hal.copy_from_elem("out", &params.out);
        eval.eval_check(
            &check,
            &[&accum, &code, &data],
            &[&mix, &out],
            params.poly_mix,
            params.po2,
            params.steps,
        );
        let mut ret = vec![H::Elem::ZERO; check.size()];
        check.view(|view| {
            ret.clone_from_slice(view);
        });
        ret
    }
}
