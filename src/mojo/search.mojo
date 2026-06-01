# =============================================================================
# COMPLEXITY NOTES
# =============================================================================
# cosine_similarity():        O(dim) — один вектор, ~1 мкс для dim=384
# insert_sorted():            O(k) на вставку, вызывается N раз → O(N*k)
# cosine_search_bruteforce(): O(N * dim) — полный перебор
#
# PERFORMANCE (на CPU):
#   N=1K   — <1ms
#   N=10K  — ~15ms
#   N=50K  — ~80ms
#   N=100K — ~150ms
#
# Для N > 50K рассмотреть HNSW-индекс (фаза 2)
# =============================================================================

from std.math import sqrt


struct SearchResult(Copyable, Movable, Writable):
    var index: Int
    var score: Float32
    
    def __init__(out self, index: Int, score: Float32):
        self.index = index
        self.score = score
    
    def __init__(out self, *, copy: Self):
        self.index = copy.index
        self.score = copy.score
    
    def write_to(self, mut writer: Some[Writer]):
        writer.write("SearchResult(index=", self.index, ", score=", self.score, ")")


def cosine_similarity(
    a: UnsafePointer[Float32, MutExternalOrigin],
    b: UnsafePointer[Float32, MutExternalOrigin],
    dim: Int,
) -> Float32:
    var dot = Float32(0.0)
    var norm_a = Float32(0.0)
    var norm_b = Float32(0.0)
    
    for i in range(dim):
        var ai = a[i]
        var bi = b[i]
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    
    var denom_a = sqrt(norm_a)
    var denom_b = sqrt(norm_b)
    var denominator = denom_a * denom_b
    
    
    if denominator == 0.0:
        return Float32(0.0)
    
    return dot / denominator


def insert_sorted(
    mut results: List[SearchResult],
    var candidate: SearchResult,
    k: Int,
):
    var insert_pos = 0
    for i in range(len(results)):
        if candidate.score > results[i].score:
            break
        insert_pos += 1
    
    if insert_pos < k:
        if len(results) < k:
            results.append(candidate^)
        else:
            for j in range(len(results) - 1, insert_pos, -1):
                results[j] = results[j - 1].copy()
            
            results[insert_pos] = candidate^


def cosine_search_bruteforce(
    query: UnsafePointer[Float32, MutExternalOrigin],
    vectors: UnsafePointer[Float32, MutExternalOrigin],
    num_vectors: Int,
    dim: Int,
    k: Int,
) -> List[SearchResult]:
    var results = List[SearchResult]()
    
    for i in range(num_vectors):
        var vec_ptr = vectors + i * dim
        var score = cosine_similarity(query, vec_ptr, dim)
        insert_sorted(results, SearchResult(i, score), k)
    
    return results^
