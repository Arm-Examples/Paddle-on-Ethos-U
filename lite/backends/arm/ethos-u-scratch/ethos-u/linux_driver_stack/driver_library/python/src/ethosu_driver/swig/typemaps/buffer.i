//
// SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

%define BUFFER_FLAG_RO 0 %enddef
%define BUFFER_FLAG_RW PyBUF_WRITABLE %enddef

%define %buffer_in(TYPEMAP, SIZE, FLAG)
    %typemap(in) (TYPEMAP, SIZE) {
      Py_buffer view;

      int res = PyObject_GetBuffer($input, &view, FLAG);
      if (res < 0) {
        PyErr_Clear();
        %argument_fail(res, "(TYPEMAP, SIZE)", $symname, $argnum);
      }

      void *buf = view.buf;
      size_t size = view.len;
      PyBuffer_Release(&view);

      $1 = ($1_ltype) buf;
      $2 = ($2_ltype) size;
    }

    %typemap(typecheck) (TYPEMAP, SIZE) {
        $1 = PyObject_CheckBuffer($input) || PyTuple_Check($input) ? 1 : 0;
    }
%enddef

%define %clear_buffer_in(TYPEMAP, SIZE)
    %typemap(in) (TYPEMAP, SIZE);
    %typemap(typecheck) (TYPEMAP, SIZE);
%enddef

%define %driver_buffer_out
    %typemap(out) (char*) {
        auto size = arg1->size();
        int readOnly = 0;
        $result = PyMemoryView_FromMemory($1, size, readOnly);
    }
%enddef

%define %clear_driver_buffer_out
    %typemap(out) (char*);
%enddef
