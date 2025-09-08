# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.


import logging
import logging
import os
from typing import cast, final, List, Optional, Union

import serializer.tosa_serializer as ts

from parser.nb.nb_graph import OpBlock, VarParam
from parser.tosa.compile_spec_schema import CompileSpec
from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.process_node import (
    process_inputs,
    process_placeholder,
    process_call_function,
    process_output
)

from paddle.lite.fbs.proto.VarType_.Type import Type


# TOSA backend debug functionality
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


class ArmCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        # TODO MLETORCH-265 Remove permute_nhwc flag
        self.permute_nhwc = False
        self.quantize_io = False
        self.tosa_version = None
        self.input_order = None

    def ethosu_compile_spec(
        self,
        config: str,
        system_config: str,
        memory_mode: str,
        extra_flags: Optional[str] = None,
        config_ini: Optional[str] = "Arm/vela.ini",
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for Ethos-U NPU

        Args:
            config: Ethos-U accelerator configuration, e.g. ethos-u55-128
            system_config: System configuration to select from the Vel
                configuration file
            memory_mode: Memory mode to select from the Vela configuration file
            extra_flags: Extra flags for the Vela compiler
            config_ini: Vela configuration file(s) in Python ConfigParser .ini
                file format
        """
        assert (
            self.output_format is None
        ), f"Output format already set to f{self.output_format}"
        self.output_format = "vela"
        self.compiler_flags = [
            f"--accelerator-config={config}",
            f"--config={config_ini}",
        ]
        if system_config is not None:
            self.compiler_flags.append(f"--system-config={system_config}")
        if memory_mode is not None:
            self.compiler_flags.append(f"--memory-mode={memory_mode}")
        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        base_tosa_version = "TOSA-0.80.0+BI"
        if "u55" in config:
            # Add the Ethos-U55 extension marker
            base_tosa_version += "+u55"
        self.tosa_version = TosaSpecification.create_from_string(base_tosa_version)

        return self

    def tosa_compile_spec(self, tosa_version: str) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for TOSA flatbuffer output
        """
        assert (
            self.output_format is None
        ), f"Output format already set: {self.output_format}"
        self.output_format = "tosa"
        self.tosa_version = TosaSpecification.create_from_string(tosa_version)
        return self

    def dump_intermediate_artifacts_to(
        self, output_path: str
    ) -> "ArmCompileSpecBuilder":
        """
        Sets a path for dumping intermediate results during such as tosa and pte.
        """
        self.path_for_intermediates = output_path
        return self

    def set_permute_memory_format(
        self, set_nhwc_permutation: bool = True
    ) -> "ArmCompileSpecBuilder":
        """
        Permute to channel last in compiler and runtime. Compilation and
        runtime will convert rank 4 inputs to channel last for each sub-graph.
        """
        self.permute_nhwc = set_nhwc_permutation
        return self

    def set_quantize_io(self, quantize_io: bool = False) -> "ArmCompileSpecBuilder":
        """
        Quantization of inputs and dequantization of outputs for cases where
        whole graph is quantized and method signature is not of quantized type.
        """
        self.quantize_io = quantize_io
        return self

    def set_input_order(
        self, input_order: Optional[str] = None
    ) -> "ArmCompileSpecBuilder":
        """
        Reorder the inputs coming in. This may be required when inputs > 1.
        And while using the U55/U85 CompileSpec.
        """
        self.input_order = input_order
        return self

    def build(self) -> List[CompileSpec]:
        """
        Generate a list of compile spec objects from the builder
        """
        assert self.tosa_version

        # Always supply a TOSA version
        self.compile_spec = [
            CompileSpec("tosa_version", str(self.tosa_version).encode())
        ]

        if self.output_format == "vela":
            self.compile_spec += [
                CompileSpec("output_format", "vela".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
            ]
        elif self.output_format == "tosa":
            self.compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        if self.path_for_intermediates is not None:
            self.compile_spec.append(
                CompileSpec("debug_artifact_path", self.path_for_intermediates.encode())
            )

        if self.permute_nhwc:
            self.compile_spec.append(
                CompileSpec("permute_memory_format", "nhwc".encode())
            )

        if self.input_order:
            self.compile_spec.append(
                CompileSpec(
                    "input_order", " ".join(map(str, self.input_order)).encode()
                )
            )

        if self.quantize_io:
            self.compile_spec.append(CompileSpec("quantize_io", "True".encode()))

        return self.compile_spec


def is_permute_memory(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "permute_memory_format":
            return spec.value.decode() == "nhwc"
    return False


def is_tosa(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "tosa"
    return False


def get_intermediate_path(compile_spec: List[CompileSpec]) -> Optional[str]:
    for spec in compile_spec:
        if spec.key == "debug_artifact_path":
            return spec.value.decode()
    return None


def _get_first_delegation_tag(graph_module) ->  Union[str, None]:
    """Get the first delegation tag from the graph_module or return None."""
    for node in graph_module.graph.nodes:
        tag = node.meta.get("delegation_tag")
        if tag:
            return tag

    logger.debug("No delegation tag found in partition.")
    return None

def get_inputs(nb_block: OpBlock):
    feed_op = nb_block.ops[0]
    # feed_var = nb_block.vars[0]

    in_names = feed_op['outputs']

    input_ops = []
    input_vars = []
    for n in in_names:
        # TODO: Support one input only
        input_ops.append(nb_block.get_op(n['arguments'][0]))
        input_vars.append((n['arguments'][0]))
    return input_ops, input_vars


def get_outputs(nb_block: OpBlock):
    fetch_op = nb_block.ops[-1]
    # fetch_var = nb_block.vars[-1]

    out_names = fetch_op['inputs']

    output_ops = []
    output_vars = []
    for n in out_names:
        # TODO: Support one input only
        output_ops.append(nb_block.get_op(n['arguments'][0]))
        output_vars.append(n['arguments'][0])

    return output_ops, output_vars

def convert(nb_block: OpBlock, params: VarParam, node_visitors, tosa_spec):
    '''  NB  to tosa
    1. feed & fetch is NB only, no need in tosa
    '''
    tosa_graph = ts.TosaSerializer("")

    in_ops, in_vars = get_inputs(nb_block)
    out_ops, out_vars = get_outputs(nb_block)

    # achieve actual param datatype
    param_dt_type_dict = {}
    for param in params:
        param_dt_type_dict[param.tensor['name']] = param.tensor['data_type']

    # process inputs
    for input in in_vars:
        process_inputs(input, tosa_graph)

    # except feed & fetch
    for var in nb_block.vars:
        if var in in_vars or var in out_vars:
            print(f"created tensor {var['name']}")
            continue
        if var['ori_name'] == "feed" or var['ori_name'] == "fetch":
            print(f"nb_block.vars check: Found feed & Fetch {var['ori_name']}")
            continue
        process_placeholder(var, tosa_graph, params)

    # except feed & fetch
    for op in nb_block.ops:
        if op in in_ops or op in out_ops:
            print(f"created op {op['name']}")
            continue
        if op['type'] == "feed" or op['type'] == "fetch":
            print(f"nb_block.ops check: Found feed & Fetch {op['type']}")
            continue
        # process call function before process output
        process_call_function(op, tosa_graph, node_visitors, tosa_spec=tosa_spec, param_dt_type_dict=param_dt_type_dict)

    # process outputs
    for out in out_vars:
        process_output(out, tosa_graph)

    return tosa_graph

