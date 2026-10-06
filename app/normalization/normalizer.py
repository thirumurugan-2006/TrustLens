from app.input.schemas import NormalizedPost
def normalize_post(post: NormalizedPost):
    if post.text: post.text=" ".join(post.text.split())
    if post.title: post.title=" ".join(post.title.split())
    return post
