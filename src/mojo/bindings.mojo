# =============================================================================
# COMPLEXITY NOTES
# =============================================================================
# search_similar():   O(N*dim + dim) — копирование данных + поиск
# save_index():       O(N*dim) — копирование всех векторов
# load_index():       O(N*dim) — загрузка всех векторов
#
# Основной overhead — конвертация Python list ↔ Mojo UnsafePointer
# =============================================================================

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import UnsafePointer

from search import cosine_search_bruteforce
from chunker import chunk_text as mojo_chunk_text


@export
def PyInit_mojorag_core() -> PythonObject:
    try:
        var mb = PythonModuleBuilder("mojorag_core")
        
        _ = mb.def_function[search_similar](
            "search_similar",
            docstring="Find k most similar vectors using cosine similarity"
        )
        _ = mb.def_function[save_index](
            "save_index",
            docstring="Save vector index to binary file"
        )
        _ = mb.def_function[load_index](
            "load_index",
            docstring="Load vector index from binary file"
        )
        _ = mb.def_function[chunk_text_py](
            "chunk_text",
            docstring="Split text into overlapping chunks"
        )
        
        return mb.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def search_similar(
    query_py: PythonObject,
    vectors_py: PythonObject,
    k_py: PythonObject,
) raises -> PythonObject:
    var k = Int(py=k_py)
    var dim = Int(py=len(query_py))
    var num_vectors = Int(py=len(vectors_py))
    
    var query_ptr = alloc[Float32](dim)
    for i in range(dim):
        query_ptr[i] = Float32(py=query_py[i])
    
    var total_size = num_vectors * dim
    var vectors_ptr = alloc[Float32](total_size)
    
    for i in range(num_vectors):
        var vec = vectors_py[i]
        for j in range(dim):
            vectors_ptr[i * dim + j] = Float32(py=vec[j])
    
    var results = cosine_search_bruteforce(query_ptr, vectors_ptr, num_vectors, dim, k)
    
    query_ptr.free()
    vectors_ptr.free()
    
    var results_list = Python.list()
    for i in range(len(results)):
        var entry = Python.dict()
        _ = entry.setitem("index", PythonObject(results[i].index))
        _ = entry.setitem("score", PythonObject(results[i].score))
        _ = results_list.append(entry)
    
    return results_list


def save_index(
    vectors_py: PythonObject,
    path_py: PythonObject,
) raises -> PythonObject:
    """
    Save vectors to binary file using Python.
    Uses struct.pack for binary serialization.
    """
    var path = String(py=path_py)
    var num_vectors = Int(py=len(vectors_py))
    
    var dim = Int(0)
    if num_vectors > 0:
        dim = Int(py=len(vectors_py[0]))
    
    # Используем Python.evaluate для struct.pack с переменным числом аргументов
    var struct_mod = Python.import_module("struct")
    var builtins = Python.import_module("builtins")
    
    # Собираем все float-значения в Python-список
    var all_values = Python.list()
    _ = all_values.append(PythonObject(num_vectors))
    _ = all_values.append(PythonObject(dim))
    
    for i in range(num_vectors):
        var vec = vectors_py[i]
        for j in range(dim):
            _ = all_values.append(vec[j])
    
    # Формируем форматную строку
    var total_floats = num_vectors * dim + 2
    var format_str = String("=") + String(total_floats) + String("f")
    
    # Используем Python.evaluate чтобы обойти ограничение на *args
    var pack_code = String("lambda fmt, vals: __import__('struct').pack(fmt, *vals)")
    var pack_fn = Python.evaluate(pack_code)
    var data = pack_fn(format_str, all_values)
    
    var file = builtins.open(path, String("wb"))
    _ = file.write(data)
    _ = file.close()
    
    return PythonObject(True)


def load_index(
    path_py: PythonObject,
) raises -> PythonObject:
    """
    Load vectors from binary file using Python.
    Uses struct.unpack for binary deserialization.
    """
    var path = String(py=path_py)
    
    var struct_mod = Python.import_module("struct")
    var builtins = Python.import_module("builtins")
    
    var file = builtins.open(path, String("rb"))
    var data = file.read()
    _ = file.close()
    
    # len() работает с PythonObject без префикса Python.
    var byte_count = len(data)
    var count = Int(byte_count) // 4
    var format_str = String("=") + String(count) + String("f")
    var unpacked = struct_mod.unpack(format_str, data)
    
    var num_vectors = Int(py=unpacked[0])
    var dim = Int(py=unpacked[1])
    
    var vectors_list = Python.list()
    var idx = 2
    for i in range(num_vectors):
        var vec_list = Python.list()
        for j in range(dim):
            _ = vec_list.append(unpacked[idx])
            idx += 1
        _ = vectors_list.append(vec_list)
    
    return vectors_list


def chunk_text_py(
    text_py: PythonObject,
    chunk_size_py: PythonObject,
    overlap_py: PythonObject,
) raises -> PythonObject:
    var text = String(py=text_py)
    var chunk_size = Int(py=chunk_size_py)
    var overlap = Int(py=overlap_py)
    
    var chunks = mojo_chunk_text(text, chunk_size, overlap)
    
    var results_list = Python.list()
    for i in range(len(chunks)):
        var entry = Python.dict()
        _ = entry.setitem("text", PythonObject(chunks[i].text))
        _ = entry.setitem("start", PythonObject(chunks[i].start))
        _ = entry.setitem("end", PythonObject(chunks[i].end))
        _ = results_list.append(entry)
    
    return results_list
