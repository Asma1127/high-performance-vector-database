from std.math import sqrt


def dot(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:
    var result = a * b
    return result.reduce_add()


def l2_norm(a: SIMD[DType.float32, 4]) -> Float32:
    var result = a * a
    return sqrt(result.reduce_add())


def cosine_similarity(
    a: SIMD[DType.float32, 4],
    b: SIMD[DType.float32, 4]
) -> Float32:
    var dot_product = dot(a, b)
    var norm_a = l2_norm(a)
    var norm_b = l2_norm(b)

    return dot_product / (norm_a * norm_b)


def main():
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](2.0, 3.0, 4.0, 5.0)

    print("Dot product:", dot(a, b))
    print("Cosine similarity:", cosine_similarity(a, b))
