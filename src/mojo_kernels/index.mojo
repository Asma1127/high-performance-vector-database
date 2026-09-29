from std.math import sqrt
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python import Python
from std.os import abort


# SIMD cosine similarity for 4-element vectors
def cosine_similarity(
    a: SIMD[DType.float32, 4],
    b: SIMD[DType.float32, 4]
) -> Float32:
    var dot_product = (a * b).reduce_add()
    var norm_a = sqrt((a * a).reduce_add())
    var norm_b = sqrt((b * b).reduce_add())

    return dot_product / (norm_a * norm_b)


# Python-callable cosine similarity
def python_cosine_similarity(
    a: PythonObject,
    b: PythonObject
) raises -> PythonObject:

    # NumPy module
    var np = Python.import_module("numpy")

    # Calculate using NumPy-compatible operations
    var dot = np.dot(a, b)
    var norm_a = np.linalg.norm(a)
    var norm_b = np.linalg.norm(b)

    return dot / (norm_a * norm_b)


# Create Python module
@export
def PyInit_index_kernel() -> PythonObject:
    try:
        var m = PythonModuleBuilder("index_kernel")

        m.def_function[python_cosine_similarity](
            "cosine_similarity"
        )

        return m.finalize()

    except e:
        abort(String("Error creating index_kernel: ", e))


