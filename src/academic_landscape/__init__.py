"""Academic landscape analyzer package."""

from .batch_extraction import BatchExtractionRunner
from .clustering import ClusterCandidate, ClusterMember, cluster_items
from .cross_paper import (
    ComparedDifference,
    ComparedStatement,
    DirectionComparison,
    compare_directions,
)
from .direction_naming import ResearchDirection, name_directions
from .embedding_backend import EmbeddingBackend, LocalEmbeddingBackend
from .evaluation import evaluate_extraction_results, percentile
from .evidence_family import (
    DatasetRecord,
    EvidenceFamily,
    EvidenceFamilyReport,
    PersonRecord,
    analyze_evidence_families,
)
from .frontier_analysis import (
    FrontierReport,
    InnovationScore,
    PopularityScore,
    build_frontier_report,
    classify_thread_frontier,
    score_innovation,
    score_popularity,
)
from .graph_projector import GraphProjectionError, GraphProjectionResult, Neo4jProjector
from .ingest import build_snapshot
from .landscape_merge import MergedFrontier, merge_frontier_signals
from .landscape_report import (
    CorpusFacts,
    LandscapeResult,
    landscape_result_to_dict,
    render_landscape_report,
)
from .models import CorpusSnapshot, DocumentRecord, PageText
from .corpus_overview import CorpusOverview, build_corpus_overview
from .paper_extraction import (
    PaperExtractionClaim,
    PaperExtractionConcept,
    PaperExtractionContext,
    PaperExtractionDataSample,
    PaperExtractionEvidence,
    PaperExtractionFramework,
    PaperExtractionMethod,
    PaperExtractionQuestion,
    PaperExtractionResult,
    PaperExtractionResultItem,
    PaperExtractionService,
)
from .pipeline import LandscapePipeline, default_cluster_count
from .publication_years import YearGuess, infer_publication_year
from .report import ReportPaths, write_report
from .retrieval import EvidenceBundle, HybridRetriever
from .retrieval_index import RetrievalIndex
from .semantic_items import SemanticItem, build_semantic_items
from .stage_reconstruction import (
    ConceptThread,
    Stage,
    StageReconstruction,
    build_concept_threads,
    reconstruct_stages,
)

__all__ = [
    "BatchExtractionRunner",
    "ClusterCandidate",
    "ClusterMember",
    "ComparedDifference",
    "ComparedStatement",
    "ConceptThread",
    "CorpusFacts",
    "CorpusOverview",
    "CorpusSnapshot",
    "DirectionComparison",
    "DocumentRecord",
    "EmbeddingBackend",
    "evaluate_extraction_results",
    "EvidenceBundle",
    "EvidenceFamily",
    "EvidenceFamilyReport",
    "FrontierReport",
    "GraphProjectionError",
    "GraphProjectionResult",
    "HybridRetriever",
    "InnovationScore",
    "LandscapePipeline",
    "LandscapeResult",
    "LocalEmbeddingBackend",
    "MergedFrontier",
    "Neo4jProjector",
    "PageText",
    "PaperExtractionClaim",
    "PaperExtractionConcept",
    "PaperExtractionContext",
    "PaperExtractionDataSample",
    "PaperExtractionEvidence",
    "PaperExtractionFramework",
    "PaperExtractionMethod",
    "PaperExtractionQuestion",
    "PaperExtractionResult",
    "PaperExtractionResultItem",
    "PaperExtractionService",
    "PopularityScore",
    "percentile",
    "ReportPaths",
    "ResearchDirection",
    "RetrievalIndex",
    "Stage",
    "StageReconstruction",
    "YearGuess",
    "build_concept_threads",
    "build_corpus_overview",
    "build_frontier_report",
    "build_semantic_items",
    "build_snapshot",
    "classify_thread_frontier",
    "default_cluster_count",
    "infer_publication_year",
    "landscape_result_to_dict",
    "merge_frontier_signals",
    "name_directions",
    "render_landscape_report",
    "score_innovation",
    "score_popularity",
    "write_report",
    "analyze_evidence_families",
    "cluster_items",
    "compare_directions",
    "DatasetRecord",
    "PersonRecord",
    "SemanticItem",
    "reconstruct_stages",
]
