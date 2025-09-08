#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
from functools import lru_cache
from itertools import chain
from pathlib import Path

from setuptools import setup
from distutils.core import Extension
from setuptools.command.build_py import build_py

logger = logging.Logger(__name__)


def linux_gcc_lib_search():
    """
    Calls the `gcc` to get linker default system paths.
    Returns:
        list of paths
    """
    cmd = 'gcc --print-search-dirs | grep libraries'
    cmd_res = os.popen(cmd).read()
    cmd_res = cmd_res.split('=')
    if len(cmd_res) > 1:
        return tuple(cmd_res[1].split(':'))
    return None


def find_includes(include_env: str = 'ETHOS_U_DRIVER_INCLUDE'):
    include_path = os.getenv(include_env, '')
    return [include_path] if include_path else ['/usr/local/include', '/usr/include']


@lru_cache(maxsize=1)
def find_driver(lib_name: str,
                optional: bool = False,
                libs_env: str = 'ETHOS_U_DRIVER_LIB',
                default_lib_search: tuple = linux_gcc_lib_search()):
    """
    Searches for library installation on the local machine.

    Args:
        lib_name: library name to find
        optional: Do not fail if optional. Default is False - fail if library was not found.
        libs_env: custom environment variable pointing to libraries location, default is 'ETHOS_U_DRIVER_LIB'
        default_lib_search: list of paths to search for a library if not found within path provided by
                            'ETHOS_U_DRIVER_LIB' env variable

    Returns:
        tuple containing name of the driver libs, paths to the libs
    """

    lib_path = os.getenv(libs_env, "")

    lib_search = [lib_path] if lib_path else default_lib_search

    libs = dict(map(lambda path: (':{}'.format(path.name), path),
                    chain.from_iterable(map(lambda lib_path: Path(lib_path).glob(lib_name),
                                            lib_search))))
    if not optional and len(libs) == 0:
        raise RuntimeError("""Ethos-U driver library {} was not found in {}. Please install driver to one of the standard
                           locations or set correct ETHOS_U_DRIVER_INCLUDE and ETHOS_U_DRIVER_LIB env variables."""
                           .format(lib_name, lib_search))

    # gives back tuple of names of the libs, set of unique libs locations and includes.
    return list(libs.keys()), list(set(
        map(lambda path: str(path.absolute().parent), libs.values())))


class LibFinderExtension(Extension):
    """
    Derived from `Extension` this class adds libraries search on the user's machine.
    SWIG options and compilation flags are updated with relevant libraries files locations (-L) and headers (-I).

    Search for the library is executed only when attributes include_dirs, library_dirs, runtime_library_dirs, libraries or
    swig_opts are queried.

    """

    def __init__(self, name, sources, libs, include_dirs=None, define_macros=None, undef_macros=None,
                 library_dirs=None,
                 libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None,
                 extra_link_args=None, export_symbols=None, language=None, **kw):
        self._include_dirs = None
        self._library_dirs = None
        self._runtime_library_dirs = None
        self._libs = libs
        super().__init__(name, sources, include_dirs, define_macros, undef_macros, library_dirs, libraries,
                         runtime_library_dirs, extra_objects, extra_compile_args, extra_link_args, export_symbols,
                         language, **kw)

    @property
    def include_dirs(self):
        return self._include_dirs + find_includes()

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs

    @property
    def library_dirs(self):
        library_dirs = self._library_dirs
        for lib in self._libs:
            _, lib_path = find_driver(lib)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @library_dirs.setter
    def library_dirs(self, library_dirs):
        self._library_dirs = library_dirs

    @property
    def runtime_library_dirs(self):
        library_dirs = self._runtime_library_dirs
        for lib in self._libs:
            _, lib_path = find_driver(lib)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @runtime_library_dirs.setter
    def runtime_library_dirs(self, runtime_library_dirs):
        self._runtime_library_dirs = runtime_library_dirs

    @property
    def libraries(self):
        libraries = self._libraries
        for lib in self._libs:
            lib_names, _ = find_driver(lib)
            libraries = libraries + lib_names

        return libraries

    @libraries.setter
    def libraries(self, libraries):
        self._libraries = libraries

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.name.__hash__()


class ExtensionPriorityBuilder(build_py):
    """
    Runs extension builder before other stages. Otherwise generated files are not included to the distribution.
    """

    def run(self):
        self.run_command('build_ext')
        return super().run()


if __name__ == '__main__':
    # mandatory extensions
    driver_module = LibFinderExtension('ethosu_driver._generated._driver',
                                       sources=['src/ethosu_driver/_generated/driver_wrap.cpp'],
                                       extra_compile_args=['-std=gnu++14'],
                                       language='c++',
                                       libs=['libethosu.a']
                                       )

    extensions_to_build = [driver_module]

    setup(
        name='ethosu_driver',
        version='1.0.0',
        author='Arm ltd',
        author_email='support@arm.com',
        description='Arm Ethos-U NPU Linux Driver Stack Python wrapper',
        url='https://git.mlplatform.org/ml/ethos-u/ethos-u-linux-driver-stack.git/',
        license='Apache License 2.0',
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: C",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        keywords=["ethos-u", "driver", "npu"],
        package_dir={'': 'src'},
        packages=[
            'ethosu_driver',
            'ethosu_driver._generated',
            'ethosu_driver._utilities'
        ],
        data_files=[('', ['LICENSE'])],
        entry_points={"console_scripts": ["inference_runner = ethosu_driver.inference_runner:main"]},
        python_requires='>=3.5',
        extras_require={"numpy": ["numpy"]},
        cmdclass={'build_py': ExtensionPriorityBuilder},
        ext_modules=extensions_to_build
    )
