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

use std::{fs, io, path::PathBuf, rc::Rc};

use clap::{Args, Parser, ValueEnum};
use risc0_zkvm::{
    get_prover_server, hashfn, ApiServer, ExecutorEnv, ExecutorImpl, ProverOpts, ProverServer,
    VerifierContext,
};

/// Runs a RISC-V ELF binary within the RISC Zero ZKVM.
#[derive(Parser)]
#[command(about, version, author)]
struct Cli {
    #[command(flatten)]
    mode: Mode,

    /// Receipt output file.
    #[arg(long)]
    receipt: Option<PathBuf>,

    /// The hash function to use to produce a proof.
    #[arg(long, value_enum, default_value_t = HashFn::Poseidon)]
    hashfn: HashFn,

    /// Whether to prove exections ending in error status.
    //
    // When false, only prove execution sessions that end in a successful
    // [ExitCode] (i.e. `Halted(0)` or `Paused(0)`. When set to true, any
    // completed execution session will be proven, including indicated
    // errors (e.g. `Halted(1)`) and sessions ending in `Fault`.
    #[arg(long)]
    prove_guest_errors: bool,

    /// File to read initial input from.
    ///
    /// Reads input from stdin if an initial input file is not provided.
    #[arg(long)]
    initial_input: Option<PathBuf>,

    /// Display verbose output.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Add environment vairables in the form of NAME=value.
    #[arg(long, action = clap::ArgAction::Append)]
    env: Vec<String>,

    /// Write "pprof" protobuf output of the guest's run to this file.
    /// You can use google's pprof (<https://github.com/google/pprof>)
    /// to read it.
    #[arg(long, env = "RISC0_PPROF_OUT")]
    pprof_out: Option<PathBuf>,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct Mode {
    #[arg(long)]
    port: Option<u16>,

    /// The ELF to execute
    #[arg(long)]
    elf: Option<PathBuf>,

    /// The image to execute
    #[arg(long)]
    image: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum HashFn {
    #[value(name = hashfn::SHA_256)]
    Sha256,
    #[value(name = hashfn::POSEIDON)]
    Poseidon,
}

pub fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    ();

    let args = Cli::parse();
    if let Some(port) = args.mode.port {
        run_server(port);
        return;
    }

    let env = {
        let mut builder = ExecutorEnv::builder();

        for var in args.env.iter() {
            let (name, value) = var
                .split_once('=')
                .expect("Environment variables should be of the form NAME=value");
            builder.env_var(name, value);
        }

        if let Some(input) = args.initial_input.as_ref() {
            builder.stdin(fs::File::open(input).unwrap());
        } else {
            builder.stdin(io::stdin());
        }

        if let Some(pprof_out) = args.pprof_out.as_ref() {
            builder.enable_profiler(pprof_out);
        }

        builder.build().unwrap()
    };

    let session = {
        let mut exec = if let Some(ref elf_path) = args.mode.elf {
            let elf_contents = fs::read(elf_path).unwrap();
            ExecutorImpl::from_elf(env, &elf_contents).unwrap()
        } else if let Some(ref image_path) = args.mode.image {
            let image_contents = fs::read(image_path).unwrap();
            let image = bincode::deserialize(&image_contents).unwrap();
            ExecutorImpl::new(env, image).unwrap()
        } else {
            unreachable!()
        };
        exec.run().unwrap()
    };

    let prover = args.get_prover();
    let ctx = VerifierContext::default();
    let receipt = prover.prove_session(&ctx, &session).unwrap();

    let receipt_data = bincode::serialize(&receipt).unwrap();
    let receipt_bytes = bytemuck::cast_slice(&receipt_data);
    if let Some(receipt_file) = args.receipt.as_ref() {
        fs::write(receipt_file, receipt_bytes).expect("Unable to write receipt file");
        if args.verbose > 0 {
            eprintln!(
                "Wrote {} bytes of receipt to {}",
                receipt_data.len(),
                receipt_file.display()
            );
        }
    }
}

impl Cli {
    fn get_prover(&self) -> Rc<dyn ProverServer> {
        let hashfn = match self.hashfn {
            HashFn::Sha256 => hashfn::SHA_256,
            HashFn::Poseidon => hashfn::POSEIDON,
        };
        let opts = ProverOpts {
            hashfn: hashfn.to_string(),
            prove_guest_errors: self.prove_guest_errors,
        };

        get_prover_server(&opts).unwrap()
    }
}

fn run_server(port: u16) {
    let addr = format!("127.0.0.1:{port}");
    let server = ApiServer::new_tcp(addr);
    server.run().unwrap()
}
