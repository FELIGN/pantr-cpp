# pantr_cxx_warnings: the warnings pantr's own targets (tests, later the bindings) compile with.
# Never exported: a consumer's warning policy is the consumer's.
include_guard(GLOBAL)

add_library(pantr_cxx_warnings INTERFACE)
if(MSVC)
  target_compile_options(pantr_cxx_warnings INTERFACE /W4 $<$<BOOL:${PANTR_WARNINGS_AS_ERRORS}>:/WX>)
else()
  # GCC's -Wconversion does not imply -Wsign-conversion (Clang's does), so both are listed.
  target_compile_options(
    pantr_cxx_warnings
    INTERFACE -Wall
              -Wextra
              -Wpedantic
              -Wconversion
              -Wsign-conversion
              -Wshadow
              -Wdouble-promotion
              $<$<BOOL:${PANTR_WARNINGS_AS_ERRORS}>:-Werror>)
endif()
