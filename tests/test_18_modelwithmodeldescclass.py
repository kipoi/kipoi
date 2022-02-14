import kipoi

def test_model():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
    
    model = kipoi.get_model("APARENT/site_probabilities", source="kipoi")

def test_model_predict():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
    # model = kipoi.get_model("APARENT/site_probabilities", source="kipoi")
    pred = model.pipeline.predict_example(batch_size=4)