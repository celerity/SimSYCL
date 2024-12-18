<p align="center">
<img src="resources/logo.png" alt="SimSYCL">
<i>“Technically correct is the best kind of correct”</i><br/>&nbsp;
</p>

# What and why is this?

SimSYCL is a single-threaded, synchronous, library-only implementation of the SYCL 2020 specification. It enables you to test your SYCL applications against simulated hardware of different characteristics and discover bugs with its extensive verification capabilities.

SimSYCL is in a very early stage of development - try it at your own risk!

## Requirements

SimSYCL requires the Boost `context` libary.

## Supported Platforms

The following platform and compiler combinations are currently tested in CI:

 * Linux with GCC 11
 * Linux with Clang 17
 * Windows with MSVC 14
 * MacOS with GCC 13

Other platforms and compilers should also work, as long as they have sufficient C++20 support.
Note that Clang versions prior to 17 do not currently work due to their incomplete CTAD support.

## Research

For a detailed introduction and evaluation of SimSYCL, please refer to the [the IWOCL'24 paper](https://dl.acm.org/doi/pdf/10.1145/3648115.3648136).

If you're using SimSYCL as part of your own research, we kindly ask you to cite it as:
```bibtex
@inproceedings{simsycl,
  title={{SimSYCL}: A {SYCL} Implementation Targeting Development, Debugging, Simulation and Conformance},
  author={Thoman, Peter and Knorr, Fabian and Crisci, Luigi},
  booktitle={Proceedings of the 12th International Workshop on {OpenCL} and {SYCL}},
  pages={1--12},
  year={2024}
}
```
