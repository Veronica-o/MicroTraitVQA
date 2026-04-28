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
    composite_score: float = 0.0
    bleu1: float = 0.0
    bleu4: float = 0.0
    rouge_l: float = 0.0
    
