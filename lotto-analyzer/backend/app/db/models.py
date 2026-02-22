import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey, ARRAY, Text, Boolean, Float, Numeric, Date
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.database import Base


class Game(Base):
    __tablename__ = "games"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    rules_json = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    draws = relationship("Draw", back_populates="game", cascade="all, delete-orphan")
    imports = relationship("Import", back_populates="game", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="game", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="game", cascade="all, delete-orphan")
    generator_profiles = relationship("GeneratorProfile", back_populates="game", cascade="all, delete-orphan")
    fraud_alerts = relationship("FraudAlert", back_populates="game", cascade="all, delete-orphan")
    jackpot_analyses = relationship("JackpotAnalysis", back_populates="game", cascade="all, delete-orphan")


class Draw(Base):
    __tablename__ = "draws"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    draw_number = Column(Integer, nullable=True)
    draw_date = Column(DateTime, nullable=False)
    numbers = Column(ARRAY(Integer), nullable=False)
    bonus_numbers = Column(ARRAY(Integer))
    raw_payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # V2.0: Emission order columns
    emission_order = Column(ARRAY(Integer), nullable=True)
    bonus_emission_order = Column(ARRAY(Integer), nullable=True)
    
    # V2.0: Jackpot columns
    jackpot_amount = Column(Numeric(15, 2), nullable=True)
    jackpot_rollover = Column(Boolean, default=False)
    jackpot_consecutive_rollovers = Column(Integer, default=0)
    must_be_won = Column(Boolean, default=False)
    n_winners_div1 = Column(Integer, nullable=True)

    game = relationship("Game", back_populates="draws")


class Import(Base):
    __tablename__ = "imports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    source = Column(String, nullable=False)
    file_hash = Column(String)
    status = Column(String, nullable=False)
    stats_json = Column(JSON)
    error_log = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    game = relationship("Game", back_populates="imports")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    name = Column(String, nullable=False)
    params_json = Column(JSON)
    results_json = Column(JSON)
    dataset_hash = Column(String)
    code_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    game = relationship("Game", back_populates="analyses")
    alerts = relationship("Alert", back_populates="analysis", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"))
    severity = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    evidence_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # V2.0: Extended changepoint columns
    affected_numbers = Column(ARRAY(Integer), nullable=True)
    magnitude = Column(Float, nullable=True)
    context_note = Column(Text, nullable=True)
    is_validated = Column(Boolean, default=False)
    validation_note = Column(Text, nullable=True)
    jackpot_at_changepoint = Column(Numeric(15, 2), nullable=True)
    emission_position_affected = Column(Integer, nullable=True)

    game = relationship("Game", back_populates="alerts")
    analysis = relationship("Analysis", back_populates="alerts")


class GeneratorProfile(Base):
    """Forensic profile of a lottery generator"""
    __tablename__ = "generator_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Analysis period
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)
    n_draws = Column(Integer, nullable=False)
    
    # Global conformity score
    conformity_score = Column(Float, nullable=True)
    conformity_ci_low = Column(Float, nullable=True)
    conformity_ci_high = Column(Float, nullable=True)
    conformity_n_simulations = Column(Integer, default=1000)
    
    # Generator type detection
    generator_type = Column(String(20), nullable=True)  # physical, rng, hybrid, unknown
    
    # Test results by category (JSONB)
    standard_tests = Column(JSONB, nullable=True)
    nist_tests = Column(JSONB, nullable=True)
    physical_tests = Column(JSONB, nullable=True)
    rng_tests = Column(JSONB, nullable=True)
    structural_tests = Column(JSONB, nullable=True)
    
    # Reproducibility
    dataset_hash = Column(String(64), nullable=False)
    app_version = Column(String(20), nullable=True)
    params = Column(JSONB, nullable=True)
    seed = Column(Integer, nullable=True)

    game = relationship("Game", back_populates="generator_profiles")


class FraudAlert(Base):
    """Fraud detection alerts"""
    __tablename__ = "fraud_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Classification
    severity = Column(String(10), nullable=False)  # INFO, WARNING, HIGH, CRITICAL
    signal_type = Column(String(50), nullable=False)
    category = Column(String(30), nullable=True)  # generator, data_quality, behavioral, structural
    
    # Description
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    statistical_evidence = Column(JSONB, nullable=True)
    
    # Affected draws
    draw_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)
    
    # Workflow
    status = Column(String(20), default='OPEN', nullable=False)  # OPEN, INVESTIGATING, CLOSED, FALSE_POSITIVE
    assigned_to = Column(String(100), nullable=True)
    resolution_note = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Link to analysis
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=True)

    game = relationship("Game", back_populates="fraud_alerts")


class JackpotAnalysis(Base):
    """Jackpot analysis results"""
    __tablename__ = "jackpot_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Test results (JSONB)
    generator_independence_test = Column(JSONB, nullable=True)
    player_bias_analysis = Column(JSONB, nullable=True)
    threshold_effect = Column(JSONB, nullable=True)
    must_be_won_analysis = Column(JSONB, nullable=True)
    jackpot_vs_emission = Column(JSONB, nullable=True)
    
    # Metadata
    n_draws_analyzed = Column(Integer, nullable=True)
    jackpot_range_min = Column(Numeric(15, 2), nullable=True)
    jackpot_range_max = Column(Numeric(15, 2), nullable=True)
    dataset_hash = Column(String(64), nullable=True)

    game = relationship("Game", back_populates="jackpot_analyses")
