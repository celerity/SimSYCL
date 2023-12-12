![simSYCL](resources/logo.png)
### The SYCL implementation you did (not) know you wanted

# Requirements
SimSYCL requires the Boost `context` libary.

# Supported Platforms
The following platform and compiler combinations are currently tested in CI:

 * Linux with GCC 11
 * Linux with Clang 17
 * Windows with MSVC 

Other platforms and compilers should also work, as long as they have sufficient C++20 support.  
Note that clang versions prior to 17 do not currently work in SimSYCL due to their CTAD limitations.

# Acknowlegments
- Fabian Knorr
- Peter Thoman
- Luigi Crisci