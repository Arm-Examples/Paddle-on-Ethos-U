from collections import defaultdict
import os
import argparse
from collections import deque

class NBDAG:
    def __init__(self, operations=None):
        self.operations = operations
        self.op_dag = self.build_op_dag_graph(True)
        self.node_to_cluster = {}

    def build_full_dag_graph(self):
        dag = defaultdict(list)
        for idx, op in enumerate(self.operations):
            inputs = op["inputs"]
            outputs = op["outputs"]
            for input_data in inputs:
                for argu in input_data['arguments']:
                    dag[argu['name']].append(f"OP_{op['type']}_{op['id']}")
            for output_data in outputs:
                for argu in output_data['arguments']:
                    dag[f"OP_{op['type']}_{op['id']}"].append(argu['name'])
        return dag

    def build_op_dag_graph(self, build_next_op:bool):
        op_dag = defaultdict(list)
        op_outputs = {}
        for idx, op in enumerate(self.operations):
            op_outputs[op['id']] = op["outputs"]

        for idx, op in enumerate(self.operations):
            op_inputs = op["inputs"]
            # Walk through all operators and find the next operator
            for op_input in op_inputs:
                for other_op_id, output in op_outputs.items():
                    outs_argus = [out['arguments'] for out in output]
                    # Case 1: each op_input node has one argument such as conv2d
                    # Case 2: op_input node has multi-arguments nodes such as concat operator
                    # The op_input arguments is in the outputs arguments
                    for op_input_arguments in op_input['arguments']:
                        if any(op_input_arguments in out_argus for out_argus in outs_argus):
                            if True == build_next_op:
                                self.operations[other_op_id]["next_op"].append(op)
                                op["prev_op"].append(self.operations[other_op_id])

                            op_dag[f"OP_{self.operations[other_op_id]['type']}_{other_op_id}"].append( \
                                f"OP_{op['type']}_{op['id']}" \
                                )

                            # print("Operator id {} is connected to operator id {}".format(other_op_id, op['id']))
        return op_dag

    def add_operation(self, op, index):
        op_id = self.operations[index]['id']
        update_id_list = self.__bfs(self.operations, op_id)
        for update_id in update_id_list:
            self.operations[update_id]["id"] = self.operations[update_id]["id"] + 1
        self.operations.insert(index, op)
        self.op_dag = self.build_op_dag_graph(False)

    def update_operation(self, op_name, inputs=None, outputs=None):
        if op_name not in self.operations:
            raise ValueError(f"Operation {op_name} does not exist.")
        if inputs is not None:
            self.operations[op_name]["input"] = inputs
        if outputs is not None:
            self.operations[op_name]["output"] = outputs
        self.full_dag = self.build_full_dag()
        self.op_dag = self.build_op_dag()

    def get_operation(self, op_name):
        if op_name not in self.operations:
            raise ValueError(f"Operation {op_name} does not exist.")
        return self.operations[op_name]

    def generate_mermaid(self, is_op_only=False):
        mermaid_lines = ["graph TD"]
        if is_op_only:
            mermaid_lines.append("    classDef opNode fill:#96f,stroke:#333,stroke-width:2px,color:#fff;")
        else:
            self.full_dag = self.build_full_dag_graph()
            mermaid_lines.append("    classDef dataNode fill:#f9f,stroke:#333,stroke-width:2px,color:#fff;")
            mermaid_lines.append("    classDef opNode fill:#96f,stroke:#333,stroke-width:2px,color:#fff;")
        dag = self.op_dag if is_op_only else self.full_dag
        for node, neighbors in dag.items():
            if is_op_only:
                mermaid_lines.append(f"    {node}({node}):::opNode")
            else:
                if node.startswith("OP"):
                    mermaid_lines.append(f"    {node}({node}):::opNode")
                else:
                    mermaid_lines.append(f"    {node}({node}):::dataNode")
            for neighbor in neighbors:
                mermaid_lines.append(f"    {node} --> {neighbor}")
        all_nodes = set(dag.keys()).union(set([neighbor for neighbors in dag.values() for neighbor in neighbors]))
        for node in all_nodes:
            if node.startswith("OP"):
                mermaid_lines.append(f"    {node}({node}):::opNode")
            else:
                mermaid_lines.append(f"    {node}({node}):::dataNode")
        return "\n".join(mermaid_lines)

    def split_subgraphs(self, dag, max_nodes=500):
        self.node_to_cluster = {}
        clusters = []
        visited = set()

        for node in dag:
            if node not in visited:
                cluster_idx = len(clusters)
                queue = deque([node])
                cluster = []
                while queue and len(cluster) < max_nodes:
                    n = queue.popleft()
                    if n not in visited:
                        cluster.append(n)
                        visited.add(n)
                        self.node_to_cluster[n] = cluster_idx
                        queue.extend(dag.get(n, []))
                        for k, v in dag.items():
                            if n in v and k not in visited:
                                queue.append(k)
                clusters.append(cluster)
        return clusters

    def generate_plantuml(self, out_dir="", max_nodes=500, is_op_only=True):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if True == is_op_only:
            dag = self.op_dag
        else:
            dag = self.build_full_dag_graph()

        clusters = self.split_subgraphs(dag, max_nodes)
        cluster_count = len(clusters)
        for cluster_id in range(cluster_count):
            subgraph_code = self.__generate_subgraph_plantuml(dag, cluster_id, is_op_only)
            with open(os.path.join(out_dir, f"subgraph_{cluster_id}.puml"), "w") as f:
                f.write(subgraph_code)

        cross_links = set()
        for node, neighbors in dag.items():
            safe_src = self.sanitize_name(node)
            src_cluster = self.find_cluster(node)
            for neighbor in neighbors:
                safe_dst = self.sanitize_name(neighbor)
                dst_cluster = self.find_cluster(neighbor)
                if src_cluster != dst_cluster:
                    cross_links.add(f"{safe_src} --> {safe_dst}")

        bridge_lines = ["@startuml"]

        bridge_lines.append("' include subgraphs")
        for i in range(cluster_count):
            bridge_lines.append(f"!include subgraph_{i}.puml")

        bridge_lines.append("\n' Main Graph connection")
        cross_links = set()
        for node, neighbors in dag.items():
            node_clean = node.replace(' ', '_')
            src_cluster = self.find_cluster(node)
            for neighbor in neighbors:
                neighbor_clean = neighbor.replace(' ', '_')
                dst_cluster = self.find_cluster(neighbor)
                if src_cluster != dst_cluster:
                    cross_links.add(f"{node_clean} --> {neighbor_clean}")

        bridge_lines.extend(sorted(cross_links))

        bridge_lines.append("@enduml")

        bridge_path = os.path.join(out_dir, "bridge.puml")
        with open(bridge_path, "w") as f:
            f.write("\n".join(bridge_lines))
        return bridge_path

    def __generate_subgraph_plantuml(self, dag, cluster_id, is_op_only):
        plantuml_lines = ["@startuml"]
        plantuml_lines.extend([
            "skinparam rectangle {",
            "    BackgroundColor #96f",
            "    FontColor white",
            "}",
            "skinparam storage {",
            "    BackgroundColor #b0b0b0",
            "    FontColor black",
            "}"
        ])

        cluster_nodes = [n for n, cid in self.node_to_cluster.items() if cid == cluster_id]

        node_defs = set()
        for node in cluster_nodes:
            safe_name = self.sanitize_name(node)
            if node.startswith("OP"):
                node_defs.add(f'rectangle "{node}" as {safe_name} [[{node}]]')
            else:
                node_defs.add(f'storage "{node}" as {safe_name}')
        plantuml_lines.extend(sorted(node_defs))

        connections = set()
        for node in cluster_nodes:
            for neighbor in dag.get(node, []):
                if self.find_cluster(neighbor) == cluster_id:
                    src = self.sanitize_name(node)
                    dst = self.sanitize_name(neighbor)
                    connections.add(f"{src} --> {dst}")
        plantuml_lines.extend(sorted(connections))

        plantuml_lines.append("@enduml")
        return "\n".join(plantuml_lines)

    def find_cluster(self, node):
        return self.node_to_cluster.get(node, -1)

    def sanitize_name(self, name):
        return (
            name.replace("/", "_")
            .replace(".", "_")
            .replace(":", "_")
            .replace("-", "_")
            .replace(" ", "__")
            .replace("[", "(").replace("]", ")")
        )


    def __dfs(self, graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        print(start, end=' ')
        for neighbor in graph[start]:
            if neighbor not in visited:
                self.__dfs(graph, neighbor, visited)
        return visited

    def __bfs(self, graph, start):
        visited = set()
        opids = []
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                opids.append(vertex)
                print(vertex, end=': ')
                neighbors = [*(graph[vertex]['next_op'])]
                for neighbor in neighbors:
                    if neighbor['id'] not in visited:
                        queue.append(neighbor['id'])
        return opids

def main():
    operations = [
        {"id":0, "type":"Conv", "inputs":[...], "outputs":[...], "prev_op":[], "next_op":[]},
        {"id":1, "type":"Relu", "inputs":[...], "outputs":[...], "prev_op":[], "next_op":[]},
    ]
    dag = NBDAG(operations)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op-only",
        action="store_true",
    )
    args = parser.parse_args()

    mermaid_code = dag.generate_mermaid(args.op_only)
    print(mermaid_code)

if __name__ == "__main__":
    main()