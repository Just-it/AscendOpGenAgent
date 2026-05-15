#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _detect_ascend_path() -> Path:
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[-1]


def _find_kernel_sources(kernel_dir: Path) -> list[Path]:
    # 新结构: op_host/*.cpp + op_kernel/*.cpp
    sources = []
    for subdir in ("op_host", "op_kernel"):
        sd = kernel_dir / subdir
        if sd.is_dir():
            sources.extend(sorted(sd.glob("*.cpp")))
    if sources:
        return sources

    # 兼容旧结构: kernel/*.cpp (排除 pybind11.cpp)
    sources = sorted(
        path for path in kernel_dir.glob("*.cpp")
        if path.name != "pybind11.cpp"
    )
    if not sources:
        raise FileNotFoundError(f"No kernel .cpp sources found in {kernel_dir}")
    return sources


def _find_pybind_or_register(kernel_dir: Path) -> Path | None:
    # 新结构: 使用 register.cpp
    reg = kernel_dir / "register.cpp"
    if reg.is_file():
        return reg
    # 旧结构: 使用 pybind11.cpp
    pyb = kernel_dir / "pybind11.cpp"
    if pyb.is_file():
        return pyb
    return None


def _extract_pybind_module_name(pybind_path: Path) -> str:
    content = pybind_path.read_text(encoding="utf-8")
    match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    if not match:
        raise ValueError(f"Unable to detect PYBIND11_MODULE name from {pybind_path}")
    return match.group(1)


def _format_cmake_list(items: list[str], indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _generate_cmakelists(
    kernel_dir: Path,
    build_dir: Path,
    module_name: str,
    sources: list[Path],
    ascend_path: Path,
    entry_path: Path,
) -> str:
    include_dirs = [kernel_dir]
    catlass_include = kernel_dir / "catlass" / "include"
    if catlass_include.is_dir():
        include_dirs.append(catlass_include)

    task_catlass_include = kernel_dir.parent / "catlass" / "include"
    if task_catlass_include.is_dir() and task_catlass_include not in include_dirs:
        include_dirs.append(task_catlass_include)

    source_lines = [str(path) for path in sources]
    include_lines = [str(path) for path in include_dirs]

    return f"""cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)

ascendc_include_directories(kernels PRIVATE
{_format_cmake_list(include_lines)}
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
)

add_library(pybind11_lib SHARED "{entry_path}")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "{kernel_dir}"
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)
"""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[build_ascendc] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build(task: str, soc_version: str, build_type: str, clean: bool) -> Path:
    task_dir = _resolve_task_dir(task)
    kernel_dir = task_dir / "kernel"
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    build_dir = kernel_dir / "build"
    ascend_path = _detect_ascend_path()
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_path)

    # Mode A — 透传: kernel/CMakeLists.txt 已存在，直接用
    native_cmake = kernel_dir / "CMakeLists.txt"
    if native_cmake.is_file():
        if clean and build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_configure = [
            "cmake",
            "-S",
            str(kernel_dir),
            "-B",
            str(build_dir),
            f"-DSOC_VERSION={soc_version}",
            f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        cmake_build = ["cmake", "--build", str(build_dir), "-j"]
        _run(cmake_configure, cwd=task_dir, env=env)
        _run(cmake_build, cwd=task_dir, env=env)
        return build_dir

    # Mode B — 自动生成: 无 CMakeLists.txt，生成后编译
    entry_path = _find_pybind_or_register(kernel_dir)
    if entry_path is None:
        raise FileNotFoundError(
            f"No entry file (register.cpp or pybind11.cpp) found in {kernel_dir}"
        )

    sources = _find_kernel_sources(kernel_dir)
    if entry_path.name == "register.cpp":
        module_name = f"{task_dir.name}_ext"
    else:
        module_name = _extract_pybind_module_name(entry_path)

    cmake_dir = build_dir / "_autogen_cmake"

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = cmake_dir / "CMakeLists.txt"
    cmakelists_path.write_text(
        _generate_cmakelists(
            kernel_dir=kernel_dir,
            build_dir=build_dir,
            module_name=module_name,
            sources=sources,
            ascend_path=ascend_path,
            entry_path=entry_path,
        ),
        encoding="utf-8",
    )

    cmake_configure = [
        "cmake",
        "-S",
        str(cmake_dir),
        "-B",
        str(build_dir),
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    cmake_build = ["cmake", "--build", str(build_dir), "-j"]

    _run(cmake_configure, cwd=task_dir, env=env)
    _run(cmake_build, cwd=task_dir, env=env)
    return build_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AscendC kernels for a task without task-local run.sh")
    parser.add_argument("task", help="Task directory name or path")
    parser.add_argument("-v", "--soc-version", default="Ascend910B2", help="Ascend SoC version")
    parser.add_argument("--build-type", default="Debug", help="CMake build type")
    parser.add_argument("--clean", action="store_true", help="Remove kernel/build before configuring")
    args = parser.parse_args()

    build_dir = build(
        task=args.task,
        soc_version=args.soc_version,
        build_type=args.build_type,
        clean=args.clean,
    )
    print(f"[build_ascendc] Build completed: {build_dir}")


if __name__ == "__main__":
    main()
