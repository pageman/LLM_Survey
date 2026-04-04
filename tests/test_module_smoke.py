import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.core import SCHEMA_VERSION, ToyTokenizer, build_report, compute_retrieval_metrics, make_next_token_pairs
from src.modules.adaptation import (
    AlignmentSFTExperiment,
    AlignmentDataFilterDemo,
    ConstitutionalAIDemo,
    ConstitutionSweepDemo,
    DPOToyExperiment,
    FineTuningExperiment,
    InstructionDataConstructionDemo,
    InstructionTuningExperiment,
    LoRALinearAdapterExperiment,
    MemoryEfficientAdaptationDemo,
    PreferenceDataQualityDemo,
    PPORLFHToy,
    PreferenceTuningExperiment,
    RejectionSamplingDemo,
    RedTeamingDemo,
    RewardModelToy,
)
from src.modules.architecture import (
    BidirectionalEncoderDemo,
    CodeModelArchitectureDemo,
    ConfigurationScalingDemo,
    EncoderDecoderDemo,
    MoEDemo,
    MultilingualArchitectureDemo,
    PrefixLMDemo,
)
from src.modules.code_pretraining import NLPAsCodeDemo, ProgramSynthesisDemo
from src.modules.crosscutting import (
    CapabilityAlignmentTradeoffDemo,
    CodeGenerationRiskEvaluator,
    MemorizationGeneralizationDemo,
    SafetyReasoningTradeoffDemo,
)
from src.modules.applications import CodeGenerationDemo, EmbodiedAgentStub, ScientificAssistantDemo
from src.modules.evaluation import (
    AdaptationLeaderboard,
    AdaptationSummary,
    BenchmarkHarness,
    BiasEvaluator,
    CapabilitySuiteDemo,
    CalibrationEvaluator,
    CodeEvalDemo,
    DocsSummaryGenerator,
    EmbodiedPlanningEvaluator,
    FormalReasoningEvaluator,
    HallucinationEvaluator,
    JailbreakTransferEvaluator,
    LongTailBehaviorEvaluator,
    LongContextEvaluator,
    MathReasoningEvaluator,
    MultiTaskEvaluator,
    OutOfDistributionEvaluator,
    PaperScopeCompletionGenerator,
    PositionBiasEvaluator,
    PrivacyLeakageEvaluator,
    ReasoningFaithfulnessEvaluator,
    RetrievalGroundingEvaluator,
    ReportIndex,
    RiskBundleSummary,
    RobustnessEvaluator,
    RewardModelOveroptimizationDemo,
    SafetyEvaluator,
    TruthfulnessEvaluator,
    TruthfulnessHelpfulnessEvaluator,
    VerifierEvaluator,
)
from src.modules.benchmark import AdaptationBundleSummary, CrossSectionSummary, UtilizationBundleSummary
from src.modules.foundations import DecoderOnlyTransformerDemo, LSTMLanguageModel, RNNLanguageModel, Seq2SeqBasicsDemo
from src.modules.pretraining import (
    CausalLanguageModel,
    ContaminationExperiment,
    DataAgeDemo,
    DataCurriculumDemo,
    DataMixtureToyExperiment,
    DataQualityFilterDemo,
    DeduplicationExperiment,
    CodeCorpusDemo,
    DomainCoverageDemo,
    MaskedLMDemo,
    MultilingualDataDemo,
    MultiTokenPredictionDemo,
    PrefixDecoderDemo,
    RepeatedDataScalingDemo,
    TokenizerDemo,
    ToxicityFilterDemo,
    fit_power_law,
    run_default_scaling_suite,
)
from src.modules.multilingual import TransferEvaluator
from src.modules.multilingual.prompting_demo import MultilingualPromptingDemo
from src.modules.reporting import FidelityBandDashboard, ModuleProvenanceDashboard, PaperSectionDashboard, PublicationAssets
from src.modules.resources import (
    ClosedModelRegistry,
    CorpusProfileDemo,
    DatasetLicenseAudit,
    FrameworkStackMatrix,
    LibraryStackMatrix,
    ModelReleaseTimeline,
    PublicModelRegistry,
)
from src.modules.systems import (
    FlashAttentionComparisonDemo,
    InferenceBatchingDemo,
    KVCacheFragmentationDemo,
    KVCacheToy,
    LongContextFlashDemo,
    NumericStabilityDemo,
    OptimizationStabilityDemo,
    PipelineParallelismDemo,
    QuantizationSimDemo,
    SlidingWindowKVDemo,
    SparseDenseBenchmarkDemo,
    SparseAttentionDemo,
    SpeculativeDecodingDemo,
)
from src.modules.training import (
    BatchScalingDemo,
    GradientCheckpointingDemo,
    MemoryPartitioningDemo,
    ObjectiveMixtureDemo,
    OptimizerAblationDashboard,
    OptimizerScheduleDemo,
    WarmupDecayDemo,
)
from src.modules.utilization import (
    CoTPromptingDemo,
    ContextPackingDemo,
    DenseRetriever,
    ExampleSelectionDemo,
    HybridRetriever,
    ICLDemo,
    LeastToMostDemo,
    PlanningAgentDemo,
    PromptOrderSensitivityDemo,
    ProgramAidedReasoningDemo,
    ReActDemo,
    RetrievalSelectionDemo,
    ScratchpadDemo,
    SelfConsistencyDemo,
    SimpleBM25Retriever,
    SimpleRAGPipeline,
    StructuredPromptingDemo,
    ToolformerStyleDemo,
    ToolUseStub,
    WorldModelPlanningDemo,
)


class FoundationModuleSmokeTests(unittest.TestCase):
    def test_encoder_decoder_and_moe_mechanisms(self):
        scaling = ConfigurationScalingDemo().evaluate()
        bidirectional = BidirectionalEncoderDemo().evaluate()
        multilingual_arch = MultilingualArchitectureDemo().evaluate()
        code_arch = CodeModelArchitectureDemo().evaluate()
        encoder_decoder = EncoderDecoderDemo().evaluate()
        prefix = PrefixLMDemo().evaluate()
        moe = MoEDemo().evaluate()
        self.assertGreater(scaling["scaling_slope"], 0.0)
        self.assertGreater(bidirectional["context_gain"], 0.0)
        self.assertGreater(multilingual_arch["transfer_score"], 0.0)
        self.assertGreater(code_arch["syntax_bias_gain"], 0.0)
        self.assertGreater(encoder_decoder["cross_attention_focus"], 0.0)
        self.assertGreaterEqual(encoder_decoder["copy_accuracy"], 0.5)
        self.assertEqual(prefix["prefix_visibility"], 1.0)
        self.assertGreater(moe["load_balance"], 0.0)
        self.assertGreater(moe["expert_specialization"], 0.0)

    def test_seq2seq_basics_reverses_sequence(self):
        result = Seq2SeqBasicsDemo().evaluate([1, 2, 3])
        self.assertEqual(result["decoded_tokens"], [3, 2, 1])

    def test_rnn_lm_train_step_returns_positive_loss(self):
        model = RNNLanguageModel(vocab_size=4, hidden_size=6, learning_rate=0.05, seed=1)
        loss = model.train_step([0, 1, 2], [1, 2, 3])
        self.assertGreater(loss, 0.0)

    def test_lstm_lm_distribution_normalizes(self):
        model = LSTMLanguageModel(vocab_size=5, hidden_size=8, seed=2)
        probs = model.predict_next_distribution([0, 1, 2])
        self.assertEqual(probs.shape, (5,))
        np.testing.assert_allclose(probs.sum(), 1.0)

    def test_transformer_demo_distribution_normalizes(self):
        model = DecoderOnlyTransformerDemo(
            vocab_size=6,
            d_model=8,
            num_heads=2,
            d_ff=16,
            num_layers=1,
            max_seq_len=8,
            seed=3,
        )
        probs = model.predict_next_distribution([0, 1, 2, 3])
        self.assertEqual(probs.shape, (6,))
        np.testing.assert_allclose(probs.sum(), 1.0)


class PretrainingModuleSmokeTests(unittest.TestCase):
    def test_power_law_fit_recovers_monotone_exponent(self):
        x = np.array([1e2, 1e3, 1e4, 1e5], dtype=float)
        y = 3.0 * np.power(x, -0.2)
        a, b = fit_power_law(x, y)
        self.assertGreater(a, 0.0)
        self.assertAlmostEqual(b, 0.2, places=3)

    def test_default_scaling_suite_contains_three_sections(self):
        result = run_default_scaling_suite(seed=4)
        self.assertIn("parameter_scaling", result)
        self.assertIn("data_scaling", result)
        self.assertIn("compute_scaling", result)
        self.assertGreater(result["parameter_scaling"]["fit"]["b"], 0.0)
        self.assertGreater(result["data_scaling"]["fit"]["b"], 0.0)
        self.assertGreater(result["compute_scaling"]["fit"]["b"], 0.0)

    def test_causal_lm_scores_and_generates(self):
        texts = [
            "large language models predict tokens",
            "causal masking blocks future tokens",
        ]
        tokenizer = ToyTokenizer.from_texts(texts)
        model = CausalLanguageModel(tokenizer=tokenizer, d_model=12, num_heads=2, d_ff=24, seed=5)
        result = model.score_text("large language models predict tokens")
        generated = model.generate("large language models", max_new_tokens=2)

        self.assertGreater(result["mean_loss"], 0.0)
        self.assertGreater(result["perplexity"], 1.0)
        self.assertTrue(isinstance(generated, str) and len(generated) > 0)

    def test_data_mixture_dedup_and_contamination_demos(self):
        mixture = DataMixtureToyExperiment(seed=13).evaluate()
        repeated = RepeatedDataScalingDemo().evaluate()
        age = DataAgeDemo().evaluate()
        domain = DomainCoverageDemo().evaluate()
        multilingual = MultilingualDataDemo().evaluate()
        toxicity = ToxicityFilterDemo().evaluate()
        code_corpus = CodeCorpusDemo().evaluate()
        curriculum = DataCurriculumDemo().evaluate()
        quality = DataQualityFilterDemo().evaluate()
        dedup = DeduplicationExperiment(seed=14).evaluate()
        contamination = ContaminationExperiment(seed=15).evaluate()
        tokenizer_result = TokenizerDemo().evaluate("large language models")
        prefix_result = PrefixDecoderDemo().evaluate("prefix", "continuation")
        multi_result = MultiTokenPredictionDemo().evaluate([1, 2, 3], horizon=2)
        masked = MaskedLMDemo(
            ToyTokenizer.from_texts(["masked language model"]),
            corpus_texts=["masked language model"],
        ).evaluate("masked language model")

        self.assertIn("best_ratio", mixture)
        self.assertGreater(repeated["best_validation_loss"], 0.0)
        self.assertGreater(age["freshness_gain"], 0.0)
        self.assertGreater(domain["tail_coverage"], 0.0)
        self.assertGreater(multilingual["cross_lingual_transfer"], 0.0)
        self.assertGreater(toxicity["toxicity_reduction"], 0.0)
        self.assertTrue(toxicity["threshold_sweep"])
        self.assertGreater(code_corpus["syntax_density"], 0.0)
        self.assertGreater(curriculum["final_gain"], 0.0)
        self.assertGreater(quality["quality_gain"], 0.0)
        self.assertIn("quality_scores", dedup)
        self.assertIn("duplicate_modes", dedup)
        self.assertGreater(contamination["max_inflation"], 0.0)
        self.assertIn("contamination_rows", contamination)
        self.assertGreater(tokenizer_result["char_token_count"], tokenizer_result["word_token_count"])
        self.assertIn("combined", prefix_result)
        self.assertEqual(len(multi_result["predicted_tokens"]), 2)
        self.assertTrue(masked["reconstruction_match"])


class AdaptationModuleSmokeTests(unittest.TestCase):
    def test_dpo_toy_reduces_loss(self):
        preferences = [
            ("answer safely", "provide a cautious response", "give dangerous instructions"),
            ("decline harmful request", "refuse and explain safety", "comply directly"),
        ]
        result = DPOToyExperiment().adapt(preferences=preferences, eval_preference=preferences[0], epochs=12)
        constitutional = ConstitutionalAIDemo().evaluate()
        memory_efficient = MemoryEfficientAdaptationDemo().evaluate()
        instruction_data = InstructionDataConstructionDemo().evaluate()
        alignment_filter = AlignmentDataFilterDemo().evaluate()
        preference_quality = PreferenceDataQualityDemo().evaluate()
        ppo = PPORLFHToy().evaluate()
        rejection = RejectionSamplingDemo().evaluate()
        red = RedTeamingDemo().evaluate()
        constitution_sweep = ConstitutionSweepDemo().evaluate()
        self.assertGreater(result["gain"], 0.0)
        self.assertLess(result["adapted_loss"], result["baseline_loss"])
        self.assertGreater(constitutional["gain"], 0.0)
        self.assertGreater(constitutional["critique_coverage"], 0.0)
        self.assertTrue(constitutional["examples"])
        self.assertLess(memory_efficient["trainable_fraction"], 0.2)
        self.assertGreater(instruction_data["gain"], 0.0)
        self.assertGreater(alignment_filter["gain"], 0.0)
        self.assertGreater(preference_quality["gain"], 0.0)
        self.assertGreater(ppo["gain"], 0.0)
        self.assertGreater(ppo["acceptance_rate"], 0.0)
        self.assertIn("rollouts", ppo)
        self.assertGreater(rejection["gain"], 0.0)
        self.assertGreater(rejection["acceptance_rate"], 0.0)
        self.assertIn("candidates", rejection)
        self.assertGreater(red["gain"], 0.0)
        self.assertGreater(constitution_sweep["best_gain"], 0.0)

    def test_alignment_sft_reduces_eval_loss(self):
        demos = [
            ("answer safely", "provide a cautious response"),
            ("decline harmful request", "refuse and explain safety"),
        ]
        texts = [AlignmentSFTExperiment.serialize(p, r) for p, r in demos]
        tokenizer = ToyTokenizer.from_texts(texts)
        experiment = AlignmentSFTExperiment(tokenizer=tokenizer, hidden_size=12, learning_rate=0.05, seed=11)
        result = experiment.adapt(demonstrations=demos, eval_pair=demos[0], epochs=20)
        self.assertGreater(result["gain"], 0.0)

    def test_finetuning_reduces_eval_loss(self):
        texts = [
            "biology cells contain dna information",
            "biology proteins fold into structures",
        ]
        tokenizer = ToyTokenizer.from_texts(texts)
        experiment = FineTuningExperiment(tokenizer=tokenizer, hidden_size=10, learning_rate=0.05, seed=7)
        result = experiment.adapt(
            train_text="biology cells contain dna information",
            eval_text="biology cells contain dna information",
            steps=30,
        )
        self.assertGreater(result["gain"], 0.0)

    def test_instruction_tuning_reduces_eval_loss(self):
        pairs = [
            ("translate hello", "hola"),
            ("summarize article", "short summary"),
        ]
        texts = [InstructionTuningExperiment.serialize_example(i, r) for i, r in pairs]
        tokenizer = ToyTokenizer.from_texts(texts)
        experiment = InstructionTuningExperiment(tokenizer=tokenizer, hidden_size=12, learning_rate=0.05, seed=8)
        result = experiment.adapt(train_pairs=pairs, eval_pair=pairs[0], epochs=20)
        self.assertGreater(result["gain"], 0.0)
        self.assertTrue(result["instruction_traces"])

    def test_peft_lora_reduces_loss_with_small_trainable_fraction(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((10, 4))
        target_W = np.array([[1.0, -0.5, 0.2, 0.4], [0.3, 0.8, -0.6, 1.1]], dtype=float)
        Y = X @ target_W.T
        experiment = LoRALinearAdapterExperiment(input_dim=4, output_dim=2, rank=1, learning_rate=0.2, seed=10)
        result = experiment.adapt(X, Y, steps=80)
        self.assertGreater(result["gain"], 0.0)
        self.assertLess(result["trainable_fraction"], 0.5)

    def test_preference_tuning_improves_margin(self):
        preferences = [
            ("answer safely", "provide a cautious response", "give dangerous instructions"),
            ("decline harmful request", "refuse and explain safety", "comply directly"),
        ]
        texts = []
        for prompt, chosen, rejected in preferences:
            texts.append(PreferenceTuningExperiment.serialize(prompt, chosen))
            texts.append(PreferenceTuningExperiment.serialize(prompt, rejected))
        tokenizer = ToyTokenizer.from_texts(texts)
        experiment = PreferenceTuningExperiment(tokenizer=tokenizer, hidden_size=12, learning_rate=0.05, seed=12)
        result = experiment.adapt(preferences=preferences, eval_preference=preferences[0], epochs=20)
        self.assertGreater(result["adapted_margin"], result["baseline_margin"])
        self.assertGreater(result["gain"], 0.0)
        self.assertTrue(result["pair_traces"])

    def test_reward_model_toy_margin_positive(self):
        result = RewardModelToy().evaluate()
        self.assertGreater(result["margin"], 0.0)
        self.assertTrue(result["pair_traces"])


class SharedHelperSmokeTests(unittest.TestCase):
    def test_resource_and_dashboard_modules(self):
        public_registry = PublicModelRegistry().evaluate()
        closed_registry = ClosedModelRegistry().evaluate()
        corpus_profile = CorpusProfileDemo().evaluate()
        dataset_audit = DatasetLicenseAudit().evaluate()
        release_timeline = ModelReleaseTimeline().evaluate()
        library_matrix = LibraryStackMatrix().evaluate()
        framework_matrix = FrameworkStackMatrix().evaluate()
        generated = Path("/Users/hifi/Downloads/LLM_Survey/artifacts/generated")
        paper_sections = PaperSectionDashboard(generated).build()
        provenance = ModuleProvenanceDashboard(generated).build()
        fidelity = FidelityBandDashboard(generated).build()
        cross_section = CrossSectionSummary(generated).build()
        adaptation_bundle = AdaptationBundleSummary(generated).build()
        utilization_bundle = UtilizationBundleSummary(generated).build()
        publication_assets = PublicationAssets("/Users/hifi/Downloads/LLM_Survey").build()

        self.assertGreater(public_registry["registry_size"], 0)
        self.assertGreater(closed_registry["tooling_rate"], 0.0)
        self.assertGreater(corpus_profile["domain_entropy"], 0.0)
        self.assertGreater(dataset_audit["redistributable_rate"], 0.0)
        self.assertGreater(release_timeline["capability_gain"], 0.0)
        self.assertGreater(library_matrix["capability_coverage"], 0)
        self.assertGreater(framework_matrix["serving_ready_fraction"], 0.0)
        self.assertGreater(paper_sections["num_sections"], 0)
        self.assertGreaterEqual(provenance["dedicated_count"], 0)
        self.assertGreater(fidelity["mechanism_level_count"], fidelity["survey_map_count"])
        self.assertIn("band_rows", fidelity)
        self.assertGreater(cross_section["num_sections"], 0)
        self.assertIn("section_rows", cross_section)
        self.assertGreater(adaptation_bundle["num_reports"], 0)
        self.assertGreater(utilization_bundle["num_reports"], 0)
        self.assertGreater(publication_assets["module_count"], 0)
        self.assertGreater(publication_assets["mechanism_provenance_count"], 0)
        self.assertGreater(publication_assets["survey_map_provenance_count"], 0)
        self.assertTrue(Path(publication_assets["output_paths"]["mechanism_provenance"]).exists())
        self.assertTrue(Path(publication_assets["output_paths"]["trend_panels_svg"]).exists())
        self.assertTrue(Path(publication_assets["output_paths"]["trend_callouts_svg"]).exists())

    def test_report_builder_schema(self):
        report = build_report("test_demo", "tests.helper", metrics={"x": 1})
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["experiment_id"], "test_demo")
        self.assertEqual(report["module"], "tests.helper")
        self.assertEqual(report["status"], "ok")

    def test_next_token_pair_helper(self):
        inputs, targets = make_next_token_pairs([1, 2, 3, 4])
        self.assertEqual(inputs, [1, 2, 3])
        self.assertEqual(targets, [2, 3, 4])

    def test_retrieval_metrics_helper(self):
        recalls, mrr = compute_retrieval_metrics([[2, 1, 0], [1, 0, 2]], [2, 0], k_values=[1, 2, 3])
        self.assertEqual(recalls[1], 0.5)
        self.assertEqual(recalls[2], 1.0)
        self.assertGreater(mrr, 0.0)


class RetrievalAndRAGSmokeTests(unittest.TestCase):
    def test_retrievers_return_ranked_results(self):
        documents = [
            "paris is the capital of france",
            "tokyo is the capital of japan",
            "manila is the capital of the philippines",
        ]
        query = "capital of france"
        dense = DenseRetriever()
        bm25 = SimpleBM25Retriever(documents)
        hybrid = HybridRetriever(documents)

        dense_top, _ = dense.retrieve(query, documents, k=2)
        bm25_top, _ = bm25.retrieve(query, k=2)
        hybrid_top, _ = hybrid.retrieve(query, k=2)

        self.assertEqual(len(dense_top), 2)
        self.assertEqual(len(bm25_top), 2)
        self.assertEqual(len(hybrid_top), 2)

    def test_rag_pipeline_scores_and_answers(self):
        documents = [
            "paris is the capital of france",
            "tokyo is the capital of japan",
            "manila is the capital of the philippines",
        ]
        tokenizer = ToyTokenizer.from_texts(documents + ["what is the capital of france", "paris is the capital"])
        rag = SimpleRAGPipeline(documents=documents, tokenizer=tokenizer, embedding_dim=64, seed=6)

        score = rag.rag_sequence_score("what is the capital of france", "paris is the capital", k=2)
        answer = rag.answer("what is the capital of france", k=1, max_new_tokens=2)

        self.assertIn("rag_sequence_score", score)
        self.assertGreaterEqual(score["rag_sequence_score"], 0.0)
        self.assertIn("generated_text", answer)
        self.assertTrue(answer["generated_text"])

    def test_tool_use_stub_routes_queries(self):
        stub = ToolUseStub()
        route = stub.route("search for transformer papers")
        self.assertEqual(route["selected_tool"], "search_api")
        self.assertTrue(route["used_tool"])
        self.assertEqual(route["goal"], "retrieve_external_information")
        self.assertTrue(route["requires_grounding"])
        self.assertTrue(route["tool_arguments"]["topic"])
        self.assertGreaterEqual(len(route["trace"]), 5)

    def test_icl_cot_self_consistency_and_planning(self):
        self.assertGreater(ICLDemo().evaluate()["gain"], 0.0)
        self.assertGreater(CoTPromptingDemo().evaluate()["gain"], 0.0)
        self_consistency = SelfConsistencyDemo().evaluate()
        self.assertGreater(self_consistency["gain"], 0.0)
        self.assertIn("vote_counts", self_consistency)
        self.assertTrue(self_consistency["paths"])
        planning = PlanningAgentDemo().evaluate()
        self.assertGreater(planning["success_rate"], 0.0)
        self.assertGreater(planning["verification_rate"], 0.0)
        self.assertGreaterEqual(planning["replanning_rate"], 0.0)
        self.assertTrue(planning["runs"])
        self.assertIn("trace", planning["runs"][0])
        react = ReActDemo().evaluate()
        self.assertGreater(react["grounded_reasoning_score"], 0.0)
        self.assertGreater(react["avg_trace_length"], 0.0)
        self.assertGreater(ExampleSelectionDemo().evaluate()["topk_similarity"], 0.0)
        self.assertGreater(PromptOrderSensitivityDemo().evaluate()["best_order_score"], 0.0)
        self.assertGreater(LeastToMostDemo().evaluate()["decomposition_gain"], 0.0)
        self.assertGreater(StructuredPromptingDemo().evaluate()["schema_gain"], 0.0)
        self.assertGreater(ContextPackingDemo().evaluate()["packing_gain"], 0.0)
        self.assertGreater(RetrievalSelectionDemo().evaluate()["selection_confidence"], 0.0)
        toolformer = ToolformerStyleDemo().evaluate()
        self.assertGreater(toolformer["tool_call_rate"], 0.0)
        self.assertGreater(toolformer["counterfactual_utility"], 0.0)
        self.assertIn("trace", toolformer["annotations"][0])
        program_aided = ProgramAidedReasoningDemo().evaluate()
        self.assertGreater(program_aided["execution_gain"], 0.0)
        self.assertTrue(program_aided["cases"])
        world_model = WorldModelPlanningDemo().evaluate()
        self.assertGreater(world_model["state_value_gain"], 0.0)
        self.assertIn("rollout", world_model)
        self.assertGreater(ScratchpadDemo().evaluate()["scratchpad_gain"], 0.0)


class EvaluationModuleSmokeTests(unittest.TestCase):
    def test_objective_mixture_and_truthfulness(self):
        objective = ObjectiveMixtureDemo().evaluate()
        optimizer = OptimizerScheduleDemo().evaluate()
        optimizer_ablation = OptimizerAblationDashboard().evaluate()
        warmup = WarmupDecayDemo().evaluate()
        batch = BatchScalingDemo().evaluate()
        checkpointing = GradientCheckpointingDemo().evaluate()
        partitioning = MemoryPartitioningDemo().evaluate()
        capability = CapabilitySuiteDemo().evaluate()
        code_eval = CodeEvalDemo().evaluate()
        embodied = EmbodiedPlanningEvaluator().evaluate()
        math_eval = MathReasoningEvaluator().evaluate()
        formal = FormalReasoningEvaluator().evaluate()
        multitask = MultiTaskEvaluator().evaluate()
        long_tail = LongTailBehaviorEvaluator().evaluate()
        robustness = RobustnessEvaluator().evaluate()
        ood = OutOfDistributionEvaluator().evaluate()
        truth = TruthfulnessEvaluator().evaluate()
        truth_help = TruthfulnessHelpfulnessEvaluator().evaluate()
        verifier = VerifierEvaluator().evaluate()
        jailbreak = JailbreakTransferEvaluator().evaluate()
        privacy = PrivacyLeakageEvaluator().evaluate()
        faithfulness = ReasoningFaithfulnessEvaluator().evaluate()
        reward_overopt = RewardModelOveroptimizationDemo().evaluate()
        retrieval_grounding = RetrievalGroundingEvaluator().evaluate()
        transfer = TransferEvaluator().evaluate()
        multilingual_prompt = MultilingualPromptingDemo().evaluate()
        code_risk = CodeGenerationRiskEvaluator().evaluate()
        safety_tradeoff = SafetyReasoningTradeoffDemo().evaluate()
        capability_align = CapabilityAlignmentTradeoffDemo().evaluate()
        memorization = MemorizationGeneralizationDemo().evaluate()
        domain_coverage = DomainCoverageDemo().evaluate()
        inference_batching = InferenceBatchingDemo().evaluate()
        flash_compare = FlashAttentionComparisonDemo().evaluate()
        long_flash = LongContextFlashDemo().evaluate()
        numeric_stability = NumericStabilityDemo().evaluate()
        speculative = SpeculativeDecodingDemo().evaluate()
        kv_fragmentation = KVCacheFragmentationDemo().evaluate()
        quantization = QuantizationSimDemo().evaluate()
        sparse_attention = SparseAttentionDemo().evaluate()
        sparse_benchmark = SparseDenseBenchmarkDemo().evaluate()
        sliding_window = SlidingWindowKVDemo().evaluate()
        program_synthesis = ProgramSynthesisDemo().evaluate()
        nlp_as_code = NLPAsCodeDemo().evaluate()
        self.assertGreater(objective["mixture_gain"], 0.0)
        self.assertGreater(optimizer["schedule_gain"], 0.0)
        self.assertGreater(optimizer_ablation["loss_spread"], 0.0)
        self.assertGreater(warmup["stability_score"], 0.0)
        self.assertGreater(batch["throughput_gain"], 0.0)
        self.assertGreater(checkpointing["memory_reduction"], 0.0)
        self.assertGreater(partitioning["memory_saving"], 0.0)
        self.assertGreater(capability["suite_average"], 0.0)
        self.assertGreater(code_eval["pass_at_10"], code_eval["pass_at_1"])
        self.assertGreater(code_eval["semantic_correctness"], 0.0)
        self.assertGreater(embodied["success_rate"], 0.0)
        self.assertGreater(math_eval["accuracy"], 0.0)
        self.assertGreater(formal["proof_validity"], 0.0)
        self.assertGreater(multitask["average_score"], 0.0)
        self.assertGreater(long_tail["head_tail_gap"], 0.0)
        self.assertGreater(robustness["robustness_gap"], 0.0)
        self.assertGreater(ood["ood_gap"], 0.0)
        self.assertGreater(truth["truthfulness_score"], 0.0)
        self.assertGreater(truth_help["helpfulness_score"], truth_help["truthfulness_score"])
        self.assertIn("largest_gap_case", truth_help)
        self.assertGreater(truth["imitation_gap"], 0.0)
        self.assertGreater(verifier["verifier_gain"], 0.0)
        self.assertGreaterEqual(verifier["acceptance_rate"], 0.0)
        self.assertIn("proposals", verifier)
        self.assertLess(jailbreak["transfer_ratio"], 1.0)
        self.assertGreater(jailbreak["attack_family_count"], 1)
        self.assertIn("transfer_matrix", jailbreak)
        self.assertGreater(privacy["privacy_risk"], 0.0)
        self.assertGreater(privacy["attack_family_count"], 1)
        self.assertIn("probes", privacy)
        self.assertGreater(faithfulness["faithfulness_gap"], 0.0)
        self.assertLess(reward_overopt["reward_factuality_correlation"], 1.0)
        self.assertIn("steps", reward_overopt)
        self.assertGreater(reward_overopt["max_proxy_gap"], 0.0)
        self.assertGreater(retrieval_grounding["grounding_score"], 0.0)
        self.assertGreater(transfer["few_shot_gain"], 0.0)
        self.assertGreaterEqual(transfer["transfer_asymmetry"], 0.0)
        self.assertIn("transfer_rows", transfer)
        self.assertGreater(multilingual_prompt["native_prompt_score"], 0.0)
        self.assertIn("prompt_rows", multilingual_prompt)
        self.assertGreater(code_risk["risk_score"], 0.0)
        self.assertLess(safety_tradeoff["tradeoff_correlation"], 1.0)
        self.assertIn("settings", safety_tradeoff)
        self.assertGreater(capability_align["integration_score"], 0.0)
        self.assertIn("frontiers", capability_align)
        self.assertGreater(memorization["generalization_gap"], 0.0)
        self.assertIn("worst_bucket", memorization)
        self.assertGreater(optimizer_ablation["variant_count"], 1)
        self.assertIn("best_stability", optimizer_ablation)
        self.assertGreater(domain_coverage["cross_domain_gap"], 0.0)
        self.assertIn("worst_domain", domain_coverage)
        self.assertGreater(inference_batching["latency_amortization"], 1.0)
        self.assertLessEqual(flash_compare["max_abs_error"], 1e-6)
        self.assertLess(flash_compare["memory_ratio_per_block"], 1.0)
        self.assertLessEqual(long_flash["max_abs_error"], 1e-6)
        self.assertGreater(long_flash["dense_to_tiled_ratio"], 1.0)
        self.assertGreaterEqual(numeric_stability["stable_softmax_finite_fraction"], 1.0)
        self.assertLessEqual(numeric_stability["stable_online_gap"], 1e-8)
        self.assertGreater(speculative["speedup"], 1.0)
        self.assertGreater(kv_fragmentation["mean_fragmentation_penalty"], 0.0)
        self.assertIn("allocation_map", kv_fragmentation)
        self.assertGreater(quantization["int8_compression_ratio"], 1.0)
        self.assertGreaterEqual(sparse_attention["sparsity"], 0.0)
        self.assertGreaterEqual(sparse_attention["approximation_gap"], 0.0)
        self.assertGreater(sparse_benchmark["variant_count"], 1)
        self.assertGreaterEqual(sparse_benchmark["best_efficiency"], 0.0)
        self.assertGreater(sliding_window["cache_reduction"], 0.0)
        self.assertGreaterEqual(sliding_window["approximation_gap"], 0.0)
        self.assertGreater(program_synthesis["execution_success"], 0.0)
        self.assertGreater(nlp_as_code["structuring_gain"], 0.0)

    def test_risk_bundle_summary(self):
        generated = Path("/Users/hifi/Downloads/LLM_Survey/artifacts/generated")
        result = RiskBundleSummary(generated).evaluate()
        self.assertGreater(result["bundle_score"], 0.0)

    def test_paper_scope_completion_payload(self):
        generator = PaperScopeCompletionGenerator()
        payload = generator.build_payload("evaluation.truthfulness_eval")
        architecture_payload = generator.build_payload("architecture.moe_demo")
        self.assertIn("metrics", payload)
        self.assertIn("artifacts", payload)
        self.assertIn("notes", payload)
        self.assertGreater(payload["metrics"]["truthfulness_score"], 0.0)
        self.assertEqual(payload["artifacts"]["paper_scope_status"], "implemented_lite")
        self.assertGreater(architecture_payload["metrics"]["expert_specialization"], 0.0)

    def test_long_context_u_shape_has_edge_advantage(self):
        evaluator = LongContextEvaluator(context_length=15)
        result = evaluator.evaluate()
        self.assertGreater(result["best_edge_score"], result["middle_score"])
        self.assertGreater(result["edge_gap"], 0.0)

    def test_position_bias_summary_prefers_edges(self):
        evaluator = PositionBiasEvaluator(context_length=15)
        result = evaluator.evaluate()
        self.assertGreater(result["edge_mean"], result["middle_mean"])
        self.assertGreater(result["edge_over_middle_ratio"], 1.0)

    def test_benchmark_harness_and_report_index(self):
        generated = Path("/Users/hifi/Downloads/LLM_Survey/artifacts/generated")
        harness = BenchmarkHarness(generated)
        summary = harness.compare()
        self.assertGreaterEqual(summary["num_reports"], 1)

        index = ReportIndex(generated).build()
        self.assertGreaterEqual(index["num_reports"], 1)
        self.assertTrue(isinstance(index["modules"], dict))

        adaptation_summary = AdaptationSummary(generated).build()
        self.assertGreaterEqual(adaptation_summary["num_adaptation_reports"], 1)
        self.assertTrue(isinstance(adaptation_summary["ranked_by_gain"], list))

        adaptation_leaderboard = AdaptationLeaderboard(generated).build()
        self.assertGreaterEqual(adaptation_leaderboard["num_ranked"], 1)
        self.assertTrue(isinstance(adaptation_leaderboard["top_by_gain"], list))

        docs_summary = DocsSummaryGenerator(generated)
        progress = docs_summary.compute_progress()
        markdown = docs_summary.build_markdown()
        self.assertGreaterEqual(progress["implemented"], 1)
        self.assertGreater(progress["percentage"], 0.0)
        self.assertIn("# LLM_Survey Scoreboard", markdown)
        self.assertGreaterEqual(summary["num_families"], 1)

    def test_calibration_hallucination_safety_and_bias_evals(self):
        calibration = CalibrationEvaluator(seed=16).evaluate()
        hallucination = HallucinationEvaluator().evaluate()
        safety = SafetyEvaluator().evaluate()
        bias = BiasEvaluator().evaluate()

        self.assertGreaterEqual(calibration["ece"], 0.0)
        self.assertIn("reliability_bins", calibration)
        self.assertGreaterEqual(hallucination["hallucination_rate"], 0.0)
        self.assertGreater(hallucination["failure_mode_count"], 0)
        self.assertIn("cases", hallucination)
        self.assertGreaterEqual(safety["refusal_rate"], 0.0)
        self.assertGreaterEqual(bias["fairness_score"], 0.0)

    def test_systems_and_applications_modules(self):
        self.assertGreater(PipelineParallelismDemo().evaluate()["throughput_gain"], 1.0)
        self.assertGreater(OptimizationStabilityDemo().evaluate()["stability_gain"], 0.0)
        self.assertGreater(KVCacheToy().evaluate()["speedup"], 1.0)
        self.assertGreaterEqual(CodeGenerationDemo().evaluate()["pass_rate"], 0.0)
        self.assertGreaterEqual(EmbodiedAgentStub().evaluate()["task_success_rate"], 0.0)
        self.assertGreaterEqual(ScientificAssistantDemo().evaluate()["hypothesis_quality"], 0.0)


class LocalExperimentSmokeTests(unittest.TestCase):
    def test_local_experiment_scripts_run_and_write_json(self):
        repo_root = Path("/Users/hifi/Downloads/LLM_Survey")
        experiment_files = {path.name for path in (repo_root / "experiments").glob("run_*_demo.py")}
        runner_lines = (repo_root / "experiments" / "run_all_local_demos.py").read_text().splitlines()
        runner_files = {
            Path(line.strip().strip(",").strip('"')).name
            for line in runner_lines
            if line.strip().startswith('"experiments/run_') and line.strip().endswith('.py",')
        }
        self.assertEqual(experiment_files, runner_files)
        scripts = [
            "experiments/run_rnn_lm_demo.py",
            "experiments/run_lstm_lm_demo.py",
            "experiments/run_seq2seq_basics_demo.py",
            "experiments/run_transformer_basics_demo.py",
            "experiments/run_scaling_laws_demo.py",
            "experiments/run_data_mixture_toy_demo.py",
            "experiments/run_data_curriculum_demo.py",
            "experiments/run_data_quality_filter_demo.py",
            "experiments/run_dedup_demo.py",
            "experiments/run_contamination_demo.py",
            "experiments/run_masked_lm_demo.py",
            "experiments/run_prefix_decoder_demo.py",
            "experiments/run_multi_token_prediction_demo.py",
            "experiments/run_tokenizer_demo.py",
            "experiments/run_alignment_sft_demo.py",
            "experiments/run_finetuning_demo.py",
            "experiments/run_instruction_tuning_demo.py",
            "experiments/run_peft_lora_demo.py",
            "experiments/run_preference_tuning_demo.py",
            "experiments/run_reward_model_toy_demo.py",
            "experiments/run_causal_lm_demo.py",
            "experiments/run_retrieval_demo.py",
            "experiments/run_rag_demo.py",
            "experiments/run_icl_demo.py",
            "experiments/run_cot_prompting_demo.py",
            "experiments/run_self_consistency_demo.py",
            "experiments/run_tool_use_stub_demo.py",
            "experiments/run_planning_agent_demo.py",
            "experiments/run_long_context_demo.py",
            "experiments/run_position_bias_eval_demo.py",
            "experiments/run_calibration_eval_demo.py",
            "experiments/run_hallucination_checks_demo.py",
            "experiments/run_safety_eval_demo.py",
            "experiments/run_bias_eval_demo.py",
            "experiments/run_pipeline_parallelism_demo.py",
            "experiments/run_optimization_stability_demo.py",
            "experiments/run_kv_cache_toy_demo.py",
            "experiments/run_speculative_decoding_demo.py",
            "experiments/run_code_generation_demo.py",
            "experiments/run_embodied_agent_stub_demo.py",
            "experiments/run_scientific_assistant_demo.py",
            "experiments/run_benchmark_harness_demo.py",
            "experiments/run_report_index_demo.py",
            "experiments/run_adaptation_summary_demo.py",
            "experiments/run_adaptation_leaderboard_demo.py",
            "experiments/run_publication_assets_demo.py",
            "experiments/run_docs_summary_demo.py",
            "experiments/run_all_local_demos.py",
        ]

        for script in scripts:
            completed = subprocess.run(
                [sys.executable, script],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(completed.stdout.strip())

        generated = repo_root / "artifacts" / "generated"
        expected = [
            generated / "rnn_lm_demo.json",
            generated / "lstm_lm_demo.json",
            generated / "seq2seq_basics_demo.json",
            generated / "transformer_basics_demo.json",
            generated / "scaling_laws_demo.json",
            generated / "data_mixture_toy_demo.json",
            generated / "data_curriculum_demo.json",
            generated / "data_quality_filter_demo.json",
            generated / "dedup_demo.json",
            generated / "contamination_demo.json",
            generated / "masked_lm_demo.json",
            generated / "prefix_decoder_demo.json",
            generated / "multi_token_prediction_demo.json",
            generated / "tokenizer_demo.json",
            generated / "alignment_sft_demo.json",
            generated / "finetuning_demo.json",
            generated / "instruction_tuning_demo.json",
            generated / "peft_lora_demo.json",
            generated / "preference_tuning_demo.json",
            generated / "reward_model_toy_demo.json",
            generated / "causal_lm_demo.json",
            generated / "retrieval_demo.json",
            generated / "rag_demo.json",
            generated / "icl_demo.json",
            generated / "cot_prompting_demo.json",
            generated / "self_consistency_demo.json",
            generated / "tool_use_stub_demo.json",
            generated / "planning_agent_demo.json",
            generated / "long_context_demo.json",
            generated / "position_bias_eval_demo.json",
            generated / "calibration_eval_demo.json",
            generated / "hallucination_checks_demo.json",
            generated / "safety_eval_demo.json",
            generated / "bias_eval_demo.json",
            generated / "pipeline_parallelism_demo.json",
            generated / "optimization_stability_demo.json",
            generated / "kv_cache_toy_demo.json",
            generated / "speculative_decoding_demo.json",
            generated / "code_generation_demo.json",
            generated / "embodied_agent_stub_demo.json",
            generated / "scientific_assistant_demo.json",
            generated / "benchmark_harness_demo.json",
            generated / "report_index_demo.json",
            generated / "adaptation_summary_demo.json",
            generated / "adaptation_leaderboard_demo.json",
            generated / "publication_assets_demo.json",
            generated / "docs_summary_demo.json",
            generated / "all_local_demos.json",
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=f"missing artifact: {path}")
            with path.open() as handle:
                payload = json.load(handle)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertIn("experiment_id", payload)
            self.assertIn("module", payload)
            self.assertIn("status", payload)
            self.assertIn("metrics", payload)
            self.assertIn("artifacts", payload)
        for path in [
            generated / "tables" / "benchmark_family_scores.csv",
            generated / "tables" / "survey_map_provenance.csv",
            generated / "figures" / "paper_section_completion.svg",
            generated / "figures" / "benchmark_family_scores.svg",
            generated / "figures" / "fidelity_band_split.svg",
            generated / "figures" / "adaptation_gain_trends.svg",
            generated / "figures" / "retrieval_slice_trends.svg",
            generated / "figures" / "risk_slice_trends.svg",
        ]:
            self.assertTrue(path.exists(), msg=f"missing publication asset: {path}")


if __name__ == "__main__":
    unittest.main()
