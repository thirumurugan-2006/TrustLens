from app.input.schemas import NormalizedPost
def normalize_post(post: NormalizedPost):
    if post.content.text: post.content.text=" ".join(post.content.text.split())
    if post.content.title: post.content.title=" ".join(post.content.title.split())
    return post
