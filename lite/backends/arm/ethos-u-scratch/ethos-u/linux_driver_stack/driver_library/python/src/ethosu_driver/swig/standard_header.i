//
// SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//
%include "stl.i"
%include "cstring.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_unordered_set.i"
%include "std_pair.i"
%include "stdint.i"
%include "carrays.i"
%include "exception.i"
%include "typemaps.i"
%include "std_iostream.i"
%include "std_shared_ptr.i"

%ignore *::operator=;
%ignore *::operator[];


// Define exception typemap to wrap exception into python exception.

%exception{
    try {
        $action
    } catch (const EthosU::Exception& e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};

%exception __getitem__ {
    try {
        $action
    } catch (const std::out_of_range &e) {
        SWIG_exception(SWIG_IndexError, const_cast<char*>(e.what()));
    } catch (const std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};

%exception __setitem__ {
    try {
        $action
    } catch (const std::out_of_range &e) {
        SWIG_exception(SWIG_IndexError, const_cast<char*>(e.what()));
    } catch (const std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};
