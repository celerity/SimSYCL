add_executable(matmul matmul.cc)

target_link_libraries(matmul PRIVATE simsycl)
set_simsycl_target_options(matmul)
