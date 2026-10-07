from app.input.pipeline import InputRouter
from app.input.schemas import NormalizedPost
from app.preprocessing.language.language_analyzer import LanguageAnalyzer
from app.validation.image_validator import validate_image
from app.validation.post_validator import validate_post
from app.validation.text_validator import validate_text
from app.validation.url_validator import validate_reddit_url, validate_platform_url


class InputPipeline:
    def __init__(self) -> None:
        self.router = InputRouter()
        self.language_analyzer = LanguageAnalyzer()
        
        try:
            from app.claims.extractor import RuleBasedClaimExtractor
            from app.claims.decomposer import RuleBasedDecomposer
            self.claim_extractor = RuleBasedClaimExtractor()
            self.claim_decomposer = RuleBasedDecomposer()
        except ImportError:
            self.claim_extractor = None
            self.claim_decomposer = None

    def process(self, input_type: str, value: str) -> NormalizedPost:
        if input_type == "text":
            validation = validate_text(value)
        elif input_type == "reddit_url":
            validation = validate_reddit_url(value)
        elif input_type == "platform_url":
            validation = validate_platform_url(value)
        elif input_type == "screenshot":
            validation = validate_image(value)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        if validation is not None and not validation.valid:
            raise ValueError(validation.errors[0])

        post = self.router.route(input_type, value)
        if "language_analysis" not in post.metadata:
            post.metadata["language_analysis"] = self.language_analyzer.analyze(post.content.text or "")

        post_validation = validate_post(post)
        if not post_validation.valid:
            raise ValueError(post_validation.errors[0])
            
        if self.claim_extractor and self.claim_decomposer:
            claims = self.claim_extractor.extract_from_post(post)
            decomposed_claims = []
            for claim in claims:
                decomposed = self.claim_decomposer.decompose(claim)
                decomposed_claims.append(decomposed.model_dump())
            post.metadata["extracted_claims"] = [c.model_dump() for c in claims]
            post.metadata["decomposed_claims"] = decomposed_claims

        return post
