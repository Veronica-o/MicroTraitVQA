from dataclasses import dataclass


@dataclass
class Figure:
    figure_id: str
    label: str
    caption: str
    image_path: str
    filename: str
    article_title: str = ""
    article_abstract: str = ""


@dataclass
class VQAResult:
    figure_id: str
    question: str
    question_type: str
    model_name: str
    answer: str
    latency_s: float
    context_used: bool
    caption_overlap: float = 0.0
    completeness: float = 0.0
    cross_model_agreement: float = 0.0
    composite_score: float = 0.0
