import os,sys
from typing import List, Optional
import argparse
import numpy as numpy

from parser.graph import ModelLoader
from parser.tosa.arm_vela import vela_compile
from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.operators.node_visitor import get_node_visitors

from parser.nb.nb_graph import ProgramStru
from parser.nb.nb_dag import NBDAG
from parser.tosa.converter import ArmCompileSpecBuilder, convert
from parser.tosa.arm_vela import vela_compile

class VelaWriter:
    def __init__(self, target, model_file, out_path, inputs, remove_op_id):

        model_file_name = "/".join(os.path.abspath(model_file).split("/")[-1:])

        if model_file_name.split(".")[-1] == "nb":
            # do whole pass.
            self.model_path = "/".join(os.path.abspath(model_file).split("/")[:-1])

            self.json_graph_name = "g_" + model_file_name.replace(".nb", ".json")
            with open(model_file, "rb") as f:
                data = f.read()

            nb_graph = ModelLoader.parser(data, "nb")
            nb_graph.parser(custom_inputs=inputs, remove_op_id=remove_op_id)
            print("parser finish")

            with open(f"{self.model_path}/{self.json_graph_name}", "w") as j:
                print(f"Write json to {self.model_path}/{self.json_graph_name}")
                j.write(nb_graph.program.to_json(indent=2))

            self.program = nb_graph.program

        elif model_file_name.split(".")[-1] == "json":
            self.program = ProgramStru.from_json(f"{model_file}")

        else:
            raise FileExistsError(f"Only support NB model or Json Graph [FILE:{model_file}]")

        self.out_path = out_path
        self.tosa_graph = None
        self.vela_model = None
        self.compile_spec = self.get_compile_spec(target=target)


    def get_compile_spec(self,
        target: str,
        intermediates: Optional[str] = None,
        reorder_inputs: Optional[str] = None,
    ) -> ArmCompileSpecBuilder:
        spec_builder = None
        if target == "TOSA":
            spec_builder = (
                ArmCompileSpecBuilder()
                .tosa_compile_spec("TOSA-0.80.0+BI")
                .set_permute_memory_format(True)
            )
        elif "ethos-u55" in target:
            spec_builder = (
                ArmCompileSpecBuilder()
                .ethosu_compile_spec(
                    target,
                    system_config="Ethos_U55_High_End_Embedded",
                    memory_mode="Shared_Sram",
                    extra_flags="--debug-force-regor --output-format=raw --verbose-operators --verbose-cycle-estimate",
                )
                .set_permute_memory_format(True)
                .set_quantize_io(True)
                .set_input_order(reorder_inputs)
            )
        elif "ethos-u85" in target:
            spec_builder = (
                ArmCompileSpecBuilder()
                .ethosu_compile_spec(
                    target,
                    system_config="Ethos_U85_SYS_DRAM_Mid",
                    memory_mode="Shared_Sram",
                    extra_flags="--output-format=raw --verbose-operators --verbose-cycle-estimate",
                )
                .set_permute_memory_format(True)
                .set_quantize_io(True)
                .set_input_order(reorder_inputs)
            )

        if intermediates is not None:
            spec_builder.dump_intermediate_artifacts_to(intermediates)

        return spec_builder.build()


    def gen_vela(self):
        # test case a add
        tosa_spec = TosaSpecification.create_from_compilespecs(self.compile_spec)

        node_visitors = get_node_visitors(tosa_spec)
        for block in self.program.blocks:
            self.tosa_graph = convert(block, self.program.params, node_visitors, tosa_spec)

        if self.tosa_graph != None:
            compile_flags = []
            for spec in self.compile_spec:
                if spec.key == "compile_flags":
                    compile_flags.append(spec.value.decode())

            # create vela
            self.vela_model = vela_compile(tosa_graph=self.tosa_graph, args=compile_flags, out_dir=self.out_path)

        return self.vela_model


    def dump_graph(self, model_file, out_path, dump_op_only=False):
        for block in self.program.blocks:
            if False == isinstance(block.dag, NBDAG):
                for block in self.program.blocks:
                    dag = NBDAG(block.ops)
                    block.dag = dag

            block.dag.generate_plantuml(f"{out_path}/out_puml/", max_nodes=7000, is_op_only=dump_op_only)


def main(model_file, out_dir, inputs, remove_op_id, is_dump_graph, dump_op_only, is_vela) :
    """Run the main entry point."""
    out_file = os.path.abspath(model_file).split("/")[-1]
    if out_file.split(".")[-1] == "nb":
        out_file = out_file.replace('.nb', '_vela.bin')
        out_file = "/".join([out_dir, out_file])
        print(f"Input model file: {model_file}, Output {out_file}")
    elif out_file.split(".")[-1] == "json":
        out_file = out_file.replace('.json', '_vela.bin')
        out_file = "/".join([out_dir, out_file])
        print(f"Input model file: {model_file}, Output {out_file}")
    else:
        raise FileExistsError(f"Only support NB model [{out_file}]")

    vela_writer = VelaWriter(target="ethos-u85-128", model_file=model_file, out_path=out_dir, inputs=inputs, remove_op_id=remove_op_id)

    if is_dump_graph:
        vela_writer.dump_graph(model_file=model_file, out_path=out_dir, dump_op_only=dump_op_only)

    if is_vela:
        vela_bin = vela_writer.gen_vela()


if __name__  == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="", required=True, type=str, help="Input NB model path")
    parser.add_argument("--out_dir", default="", required=True, type=str, help="Output Vela binary path")

    parser.add_argument("--inputs", type=str, help="input tensor infer [NAME SHAPE] e.g. input 1,3,224,224")
    parser.add_argument("--remove_op_id", default="-1", type=int, help="Remove Op of id and last, -1 is not remove")
    parser.add_argument("--dump_graph", default=False, help="dump Puml file for json graph")
    parser.add_argument('--do_vela', action='store_true', help="do with vela output")
    parser.add_argument("--op_only", default=True, help="dump full mermaid mmd file for json graph")

    args = parser.parse_args()
    print(f"Do vela {args.do_vela}")

    main(args.model_path, args.out_dir, args.inputs, args.remove_op_id, args.dump_graph, args.op_only, args.do_vela)

