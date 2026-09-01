# Contributing to Dṛṣṭi

Thanks for your interest in contributing! Dṛṣṭi is a young project – every
patch, design note, and bug report moves it forward.

## 1. Picking something to work on

A good first step:

* **Bug reports.** Open an issue with a minimal reproducer, your OS /
  compiler / CMake versions, and `cmake --build build -v` output.
* **Small cleanups.** Typos, `-W` warnings, missing `[[nodiscard]]`, missing
  tests, better doc comments – all welcome.
* **Module stubs.** Any of `analysis`, `provenance`, `profiling`,
  `diagnosis`, `optimizer`, or `backends` is a good place to add a first
  real algorithm (see `include/drishti/<module>/` for the public API shape).

## 2. Setting up

```bash
git clone <your fork> && cd drishti
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DDRISHTI_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

If you changed any C++ files, please format them:

```bash
clang-format -i $(find include lib tools tests -name '*.h' -o -name '*.cpp')
```

## 3. Coding conventions

* **Language.** C++20. No compiler-specific extensions beyond the ones
  already permitted by the top-level `CMakeLists.txt`.
* **Headers.** Every public header under `include/drishti/` is
  self-contained (must compile as first include) and uses
  `#pragma once`-equivalent `#ifndef` / `#define` guards.
* **Naming.** `snake_case` for variables, functions, and files;
  `PascalCase` for classes, concepts, and type aliases; namespace
  `drishti::<module>` (e.g. `drishti::analysis`).
* **Ownership.** Prefer value semantics, `std::unique_ptr`, and
  `std::span` / `std::string_view` over raw owning pointers.
* **Tests.** New public API must ship with a matching GoogleTest suite
  under `tests/<module>/`. Register suites with `gtest_discover_tests`.
* **CMake.** Each module under `lib/<name>` produces a static library
  `drishti_<name>` with an alias `drishti::<name>` and links PUBLICly to
  `drishti_common` plus exactly the modules it actually needs.

## 4. Pull request checklist

- [ ] `cmake --build build` succeeds with your compiler in `Debug` and `Release`.
- [ ] `ctest --test-dir build` passes (add new tests under `tests/<module>/`).
- [ ] `clang-format` is applied (see `.clang-format`).
- [ ] No hardcoded machine-specific paths. Prefer CMake configure-time
      discovery (`find_package`, `FetchContent`, etc.).
- [ ] The PR description states **what** and **why** and references any
      related issues.

## 5. Getting help

Open a GitHub issue or draft PR with `[WIP]` in the title; design discussion
is very welcome before a code-heavy implementation.

---

## Conduct

Be kind. Be precise. Disagree on technical merit, never on people. Anything
that would make a reasonable person uncomfortable has no place here –
the maintainers will enforce that.
