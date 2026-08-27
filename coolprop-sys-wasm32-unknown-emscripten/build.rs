use std::{
    env, fs,
    path::{Path, PathBuf},
};

const LIB_PREFIX: &str = "lib";

#[cfg(feature = "static-refprop")]
const LIB_NAME: &str = "CoolProp-refprop";
#[cfg(not(feature = "static-refprop"))]
const LIB_NAME: &str = "CoolProp";

const LIB_EXTENSION: &str = ".so";
const STATIC_LIB_EXTENSION: &str = ".a";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap().to_lowercase();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap().to_lowercase();

    if target_os == "emscripten" && target_arch == "wasm32" {
        println!("cargo:rustc-link-lib=c++abi");
        println!("cargo:rustc-link-lib=c++");

        let src_dir = setup_src_dir();
        if cfg!(feature = "static-link") {
            setup_static_lib(&src_dir);
        } else {
            let dylib_file_name = format!("{}{}{}", LIB_PREFIX, LIB_NAME, LIB_EXTENSION);
            let lib_path = PathBuf::from("lib").join(&dylib_file_name).canonicalize().expect("bundled dylib should exist");
            
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let target_lib = out_dir.join(&dylib_file_name);
            fs::copy(&lib_path, &target_lib).unwrap();

            println!("cargo:wasm_path={}", target_lib.display());
        }
    }
}

fn setup_src_dir() -> PathBuf {
    PathBuf::from("lib").canonicalize().expect("bundled CoolProp `lib` directory should exist")
}

fn setup_static_lib(src_dir: &Path) {
    let file_name = format!("{}{}{}", LIB_PREFIX, LIB_NAME, STATIC_LIB_EXTENSION);
    let static_dir = src_dir.join("static");
    let lib_path = static_dir.join(&file_name);
    
    assert!(
        lib_path.exists(),
        "expected static CoolProp library at {}",
        lib_path.display()
    );

    println!("cargo:rustc-link-search=native={}", static_dir.to_str().unwrap());
    println!("cargo:rustc-link-lib=static={}", LIB_NAME);
}