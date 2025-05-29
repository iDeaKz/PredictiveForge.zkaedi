import pytest

def test_basic_addition():
    """Test basic integer addition."""
    assert 1 + 1 == 2

@pytest.mark.parametrize("a,b,expected", [
    ("Py", "thon", "Python"),
    ("", "abc", "abc"),
    ("abc", "", "abc"),
    ("", "", ""),
])
def test_string_concatenation(a, b, expected):
    """Test string concatenation with various cases."""
    assert a + b == expected

def test_list_append_and_type():
    """Test appending to a list and type check."""
    lst = []
    lst.append(42)
    assert lst == [42]
    assert isinstance(lst, list)

def test_int_division():
    """Test integer division and ZeroDivisionError."""
    assert 10 // 2 == 5
    with pytest.raises(ZeroDivisionError):
        _ = 1 // 0

def test_string_strip_and_empty():
    """Test string strip and empty string."""
    s = "  hello  "
    assert s.strip() == "hello"
    assert "".strip() == ""

def test_list_index_and_value_error():
    """Test list index and ValueError for missing element."""
    items = [1, 2, 3]
    assert items.index(2) == 1
    with pytest.raises(ValueError):
        items.index(99)

def test_dict_get_default():
    """Test dict get with and without default."""
    d = {"a": 1}
    assert d.get("a") == 1
    assert d.get("b") is None
    assert d.get("b", 42) == 42

def test_set_add_and_len():
    """Test set add and length."""
    s = set()
    s.add(5)
    assert 5 in s
    assert len(s) == 1

def test_tuple_immutability():
    """Test tuple immutability."""
    t = (1, 2, 3)
    with pytest.raises(AttributeError):
        t.append(4)
    with pytest.raises(TypeError):
        t[0] = 10

def test_float_rounding():
    """Test float rounding."""
    assert round(2.675, 2) == 2.67 or round(2.675, 2) == 2.68  # Python float rounding quirk

def test_dict_comprehension():
    """Test dict comprehension."""
    d = {x: x * x for x in range(3)}
    assert d == {0: 0, 1: 1, 2: 4}

def test_list_comprehension():
    """Test list comprehension."""
    lst = [x * 2 for x in range(4)]
    assert lst == [0, 2, 4, 6]

def test_set_comprehension():
    """Test set comprehension."""
    s = {x for x in [1, 2, 2, 3]}
    assert s == {1, 2, 3}

def test_string_join_and_split():
    """Test string join and split."""
    items = ["a", "b", "c"]
    s = "-".join(items)
    assert s == "a-b-c"
    assert s.split("-") == items

def test_dict_lookup_and_missing_key():
    """Test dict lookup and missing key raises KeyError."""
    d = {"a": 1, "b": 2}
    assert d["b"] == 2
    with pytest.raises(KeyError):
        _ = d["missing"]

def test_set_membership_and_empty():
    """Test set membership and empty set."""
    s = {1, 2, 3}
    assert 2 in s
    empty = set()
    assert 1 not in empty

def test_tuple_unpacking_and_length():
    """Test tuple unpacking and length."""
    a, b = (5, 10)
    assert a == 5 and b == 10
    t = (1, 2, 3, 4)
    assert len(t) == 4

def test_float_comparison_precision():
    """Test float comparison with precision tolerance."""
    result = 0.1 + 0.2
    assert abs(result - 0.3) < 1e-9

def test_dict_update_and_keys():
    """Test dict update and keys."""
    d = {"x": 1}
    d.update({"y": 2})
    assert d == {"x": 1, "y": 2}
    assert set(d.keys()) == {"x", "y"}

def test_remove_from_list_and_value_error():
    """Test removing from list and ValueError on missing element."""
    items = [1, 2, 3]
    items.remove(2)
    assert items == [1, 3]
    with pytest.raises(ValueError):
        items.remove(99)

def test_reverse_list_and_type():
    """Test reversing a list and type check."""
    items = [1, 2, 3]
    items.reverse()
    assert items == [3, 2, 1]
    assert isinstance(items, list)

def test_uppercase_string_and_type():
    """Test string uppercasing and type."""
    s = "python"
    result = s.upper()
    assert result == "PYTHON"
    assert isinstance(result, str)

def test_set_difference_and_union():
    """Test set difference and union."""
    a = {1, 2, 3}
    b = {2, 3, 4}
    assert a - b == {1}
    assert a | b == {1, 2, 3, 4}

def test_list_extend_and_empty():
    """Test list extend with another list and with empty."""
    a = [1, 2]
    b = [3, 4]
    a.extend(b)
    assert a == [1, 2, 3, 4]
    c = []
    a.extend(c)
    assert a == [1, 2, 3, 4]

def test_dict_pop_and_keyerror():
    """Test dict pop and KeyError on missing key."""
    d = {"x": 1, "y": 2}
    val = d.pop("x")
    assert val == 1 and "x" not in d
    with pytest.raises(KeyError):
        d.pop("notfound")

@pytest.mark.parametrize("container,expected_type", [
    ([1, 2, 3], list),
    ((1, 2, 3), tuple),
    ({1, 2, 3}, set),
    ({"a": 1}, dict),
    ("abc", str),
])
def test_type_checking(container, expected_type):
    """Test type checking for various containers."""
    assert isinstance(container, expected_type)

def test_none_and_empty_inputs():
    """Test handling of None and empty containers."""
    assert not bool([])
    assert not bool({})
    assert not bool(set())
    assert not bool("")
    assert None is None

def test_large_numbers():
    """Test operations with large numbers and infinity."""
    big = 10**100
    assert big + 1 > big
    assert float('inf') > 1e308

def test_unicode_string_operations():
    """Test string operations with Unicode characters."""
    s = "café"
    assert s.upper() == "CAFÉ"
    assert "é" in s

def test_immutable_type_errors():
    """Test that immutable types raise errors on mutation."""
    t = (1, 2, 3)
    with pytest.raises(AttributeError):
        t.append(4)
    with pytest.raises(TypeError):
        t[0] = 99

import copy

def test_copy_behavior():
    """Test shallow vs deep copy behavior."""
    a = [1, [2, 3]]
    b = copy.copy(a)
    c = copy.deepcopy(a)
    a[1][0] = 99
    assert b[1][0] == 99  # shallow copy affected
    assert c[1][0] == 2   # deep copy not affected

def test_sort_and_reverse_edge_cases():
    """Test sorting and reversing edge cases."""
    lst = []
    lst.sort()
    assert lst == []
    lst = [1]
    lst.reverse()
    assert lst == [1]

def test_type_casting():
    """Test explicit type conversions."""
    assert int("42") == 42
    assert str(42) == "42"
    assert list("abc") == ["a", "b", "c"]

def test_exception_message():
    """Test exception messages."""
    with pytest.raises(ValueError, match="invalid literal"):
        int("not_a_number")

def test_chained_operations():
    """Test chaining multiple operations."""
    assert "-".join(sorted(["b", "a", "c"])) == "a-b-c"

def test_boolean_short_circuit():
    """Test boolean short-circuit logic."""
    x = None
    assert (x is None or x > 0)
