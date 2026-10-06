from app.pipeline.input_pipeline import InputPipeline


pipeline = InputPipeline()
post = pipeline.process(
    "screenshot",
    "test_image.png",
)

print()
print("========== TRUSTLENS SCREENSHOT PIPELINE ==========")
print()
print("Post ID:", post.post_id)
print("Platform:", post.platform)
print("OCR Text:", post.text)
print("Images:", post.images)
print("Comments:", post.comments)
print("Metadata:", post.metadata)
print()
print("JSON should be saved under:")
print("data/processed/normalized/screenshots/")
print()
print("===================================================")
