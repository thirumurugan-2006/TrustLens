from app.pipeline.input_pipeline import InputPipeline


pipeline = InputPipeline()

post = pipeline.process(
    "text",
    """
    Congratulations!
    You have been selected for a work-from-home job.
    Earn ₹50,000 every month.
    Pay ₹500 registration fee.
    """
)

print("\n========== TRUSTLENS INPUT PIPELINE ==========\n")

print("Post ID:", post.post_id)
print("Platform:", post.platform)
print("Text:", post.content.text)
print("Images:", post.media.images)
print("Comments:", post.comments)
print("Metadata:", post.metadata)

print("\n==============================================\n")