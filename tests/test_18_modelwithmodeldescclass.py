import kipoi

def test_model():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
