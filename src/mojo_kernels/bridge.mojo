from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python import Python
from std.os import abort


def cosine_similarity(
    a: PythonObject,
    b: PythonObject
) raises -> PythonObject:
    var np = Python.import_module("numpy")

    var dot = np.dot(a, b)
    var norm_a = np.linalg.norm(a)
    var norm_b = np.linalg.norm(b)

    return dot / (norm_a * norm_b)


@export
def PyInit_bridge() -> PythonObject:
    try:
        var m = PythonModuleBuilder("bridge")
        m.def_function[cosine_similarity]("cosine_similarity")
        return m.finalize()
    except e:
        abort(String("Error creating bridge: ", e))
