// Test file to demonstrate macro expansion debugging

// Include your macro (adjust path as needed)
// use teeny_core::expr;
// use teeny_core::expr_debug;

fn main() {
    // Test the macro expansion
    let a = 1;
    let b = 2;
    let c = 3;

    // This will show you the tokens being processed
    // expr_debug!(a.b * c);

    // You can also use this to see the raw tokens:
    println!("Raw tokens: {:?}", stringify!(a.b * c));

    // To see what the macro expands to, you can use:
    // cargo rustc -- -Z unstable-options --pretty=expanded
    // Or install cargo-expand and use: cargo expand
}

// Alternative: Use compile-time debugging with compile_error!
#[macro_export]
macro_rules! debug_tokens {
    ($($tokens:tt)*) => {
        compile_error!(concat!("Tokens: ", stringify!($($tokens)*)));
    };
}

// Uncomment this to see tokens at compile time:
// debug_tokens!(a.b * c);
