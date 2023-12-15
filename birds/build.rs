use std::env;
use std::path::{Path};

fn main() {
    let pwd_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = Path::new(&*pwd_dir).join("cuda"); //directory of the .so file
    println!("cargo:rustc-link-search=native={}", path.to_str().unwrap());
    println!("cargo:rustc-link-lib=dylib=audio_wombat"); //name of the .so file (minus the lib)
    // println!("cargo:rustc-link-lib=static=add");
    // println!("cargo:rerun-if-changed=src/hello.c");
}
