"""
Performance Tracking and Success Rate Measurement System

Provides real-time analytics, trend tracking, and comprehensive
performance measurement for Phase 4.3 adaptive behavior.
"""

import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from .executor import ExecutionResult, ExecutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    object_name: str
    object_category: str
    success: bool
    grasp_confidence: float
    execution_time: float
    gripper_force: float
    lift_height: float
    stability_score: float
    contact_points: int
    retry_count: int
    error_message: Optional[str] = None

@dataclass
class PerformanceWindow:
    """Performance metrics over a time window."""
    window_start: float
    window_end: float
    total_attempts: int
    total_successes: int
    success_rate: float
    avg_execution_time: float
    avg_force: float
    avg_stability: float
    category_breakdown: Dict[str, float]
    trend_direction: str  # "improving", "declining", "stable"

@dataclass
class PerformanceTrends:
    """Long-term performance trends and analysis."""
    overall_trend: str
    success_rate_trend: List[float]
    execution_time_trend: List[float]
    force_trend: List[float]
    stability_trend: List[float]
    category_trends: Dict[str, str]
    improvement_rate: float  # % improvement per hour
    
    
class PerformanceTracker:
    """
    Real-time performance tracking and analytics system.
    
    Tracks success rates, execution metrics, and performance trends
    with rolling windows and adaptive thresholds.
    """
    
    def __init__(self, 
                 window_size_minutes: int = 30,
                 min_samples_for_trend: int = 10):
        """
        Initialize performance tracker.
        
        Args:
            window_size_minutes: Size of rolling window for trend analysis
            min_samples_for_trend: Minimum samples needed for trend analysis
        """
        self.window_size = window_size_minutes * 60  # Convert to seconds
        self.min_samples = min_samples_for_trend
        
        # Performance data storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.session_start_time = time.time()
        
        # Performance thresholds (can be adapted based on baseline)
        self.success_rate_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'acceptable': 0.4,
            'poor': 0.2
        }
        
        # Trend detection parameters
        self.trend_sensitivity = 0.05  # 5% change for trend detection
        
        # Results directory
        self.results_dir = Path("results/performance_tracking")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Performance tracker initialized")
    
    def record_attempt(self, 
                      result: ExecutionResult,
                      object_name: str,
                      object_category: str,
                      grasp_confidence: float = 0.0) -> None:
        """
        Record a grasp attempt with detailed metrics.
        
        Args:
            result: Execution result from grasp attempt
            object_name: Name of object being grasped
            object_category: Category of object (Can, Box, etc.)
            grasp_confidence: Confidence of grasp prediction
        """
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            object_name=object_name,
            object_category=object_category,
            success=result.success,
            grasp_confidence=grasp_confidence,
            execution_time=result.metrics.execution_time if result.metrics else 0.0,
            gripper_force=result.gripper_force,
            lift_height=result.lift_height_achieved,
            stability_score=result.metrics.grasp_stability_score if result.metrics else 0.0,
            contact_points=result.metrics.contact_points if result.metrics else 0,
            retry_count=result.retry_count,
            error_message=result.error_message
        )
        
        self.metrics_history.append(metrics)
        
        # Log significant events
        if result.success:
            logger.info(f"✅ Success: {object_name} ({object_category}) - "
                       f"Force: {result.gripper_force:.1f}N, "
                       f"Lift: {result.lift_height_achieved:.3f}m")
        else:
            logger.warning(f"❌ Failed: {object_name} ({object_category}) - "
                          f"{result.error_message or 'Unknown failure'}")
    
    def get_current_session_stats(self) -> Dict:
        """Get statistics for current session."""
        if not self.metrics_history:
            return self._empty_stats()
        
        total_attempts = len(self.metrics_history)
        successes = sum(1 for m in self.metrics_history if m.success)
        success_rate = successes / total_attempts
        
        # Category breakdown
        category_stats = defaultdict(list)
        for m in self.metrics_history:
            category_stats[m.object_category].append(m.success)
        
        category_rates = {
            cat: sum(successes) / len(successes) if successes else 0.0
            for cat, successes in category_stats.items()
        }
        
        # Performance averages (successful attempts only)
        successful_metrics = [m for m in self.metrics_history if m.success]
        
        avg_execution_time = (np.mean([m.execution_time for m in successful_metrics]) 
                            if successful_metrics else 0.0)
        avg_force = (np.mean([m.gripper_force for m in successful_metrics]) 
                    if successful_metrics else 0.0)
        avg_stability = (np.mean([m.stability_score for m in successful_metrics]) 
                        if successful_metrics else 0.0)
        avg_lift = (np.mean([m.lift_height for m in successful_metrics]) 
                   if successful_metrics else 0.0)
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration_minutes': session_duration / 60,
            'total_attempts': total_attempts,
            'total_successes': successes,
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'avg_force': avg_force,
            'avg_stability': avg_stability,
            'avg_lift_height': avg_lift,
            'category_breakdown': category_rates,
            'performance_rating': self._rate_performance(success_rate),
            'attempts_per_minute': total_attempts / (session_duration / 60) if session_duration > 0 else 0.0
        }
    
    def get_rolling_window_stats(self) -> PerformanceWindow:
        """Get statistics for recent time window."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Filter to recent metrics
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= window_start
        ]
        
        if not recent_metrics:
            return PerformanceWindow(
                window_start=window_start,
                window_end=current_time,
                total_attempts=0,
                total_successes=0,
                success_rate=0.0,
                avg_execution_time=0.0,
                avg_force=0.0,
                avg_stability=0.0,
                category_breakdown={},
                trend_direction="stable"
            )
        
        total_attempts = len(recent_metrics)
        total_successes = sum(1 for m in recent_metrics if m.success)
        success_rate = total_successes / total_attempts
        
        # Category breakdown for window
        category_stats = defaultdict(list)
        for m in recent_metrics:
            category_stats[m.object_category].append(m.success)
        
        category_breakdown = {
            cat: sum(successes) / len(successes) if successes else 0.0
            for cat, successes in category_stats.items()
        }
        
        # Averages for successful attempts in window
        successful_window = [m for m in recent_metrics if m.success]
        avg_execution_time = (np.mean([m.execution_time for m in successful_window]) 
                            if successful_window else 0.0)
        avg_force = (np.mean([m.gripper_force for m in successful_window]) 
                    if successful_window else 0.0)
        avg_stability = (np.mean([m.stability_score for m in successful_window]) 
                        if successful_window else 0.0)
        
        # Trend detection
        trend_direction = self._detect_trend_direction()
        
        return PerformanceWindow(
            window_start=window_start,
            window_end=current_time,
            total_attempts=total_attempts,
            total_successes=total_successes,
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            avg_force=avg_force,
            avg_stability=avg_stability,
            category_breakdown=category_breakdown,
            trend_direction=trend_direction
        )
    
    def analyze_performance_trends(self) -> PerformanceTrends:
        """Analyze long-term performance trends."""
        if len(self.metrics_history) < self.min_samples:
            return PerformanceTrends(
                overall_trend="insufficient_data",
                success_rate_trend=[],
                execution_time_trend=[],
                force_trend=[],
                stability_trend=[], 
                category_trends={},
                improvement_rate=0.0
            )
        
        # Calculate trends using sliding windows
        window_size = max(5, len(self.metrics_history) // 10)  # Adaptive window size
        
        success_rates = []
        execution_times = []
        forces = []
        stabilities = []
        
        for i in range(window_size, len(self.metrics_history)):
            window = self.metrics_history[i-window_size:i]
            
            # Success rate for this window
            successes = sum(1 for m in window if m.success)
            success_rates.append(successes / len(window))
            
            # Average metrics for successful attempts in window
            successful = [m for m in window if m.success]
            if successful:
                execution_times.append(np.mean([m.execution_time for m in successful]))
                forces.append(np.mean([m.gripper_force for m in successful]))
                stabilities.append(np.mean([m.stability_score for m in successful]))
            else:
                execution_times.append(0.0)
                forces.append(0.0)
                stabilities.append(0.0)
        
        # Determine overall trend
        overall_trend = self._determine_overall_trend(success_rates)
        
        # Category trends
        category_trends = {}
        categories = set(m.object_category for m in self.metrics_history)
        for category in categories:
            cat_metrics = [m for m in self.metrics_history if m.object_category == category]
            if len(cat_metrics) >= 5:
                cat_success_rates = []
                cat_window_size = max(3, len(cat_metrics) // 5)
                for i in range(cat_window_size, len(cat_metrics)):
                    window = cat_metrics[i-cat_window_size:i]
                    successes = sum(1 for m in window if m.success)
                    cat_success_rates.append(successes / len(window))
                category_trends[category] = self._determine_overall_trend(cat_success_rates)
        
        # Calculate improvement rate (% per hour)
        improvement_rate = self._calculate_improvement_rate(success_rates)
        
        return PerformanceTrends(
            overall_trend=overall_trend,
            success_rate_trend=success_rates,
            execution_time_trend=execution_times,
            force_trend=forces,
            stability_trend=stabilities,
            category_trends=category_trends,
            improvement_rate=improvement_rate
        )
    
    def print_realtime_dashboard(self) -> None:
        """Print real-time performance dashboard."""
        print("\\n" + "="*50)
        print("📊 REAL-TIME PERFORMANCE DASHBOARD")
        print("="*50)
        
        # Current session stats
        session_stats = self.get_current_session_stats()
        
        print(f"\\n⏱️ SESSION OVERVIEW:")
        print(f"   Duration: {session_stats['session_duration_minutes']:.1f} minutes")
        print(f"   Attempts: {session_stats['total_attempts']} "
              f"({session_stats['attempts_per_minute']:.1f}/min)")
        print(f"   Success Rate: {session_stats['success_rate']:.1%} "
              f"({session_stats['performance_rating']})")
        
        print(f"\\n⚡ PERFORMANCE METRICS:")
        print(f"   Avg Execution Time: {session_stats['avg_execution_time']:.1f}s")
        print(f"   Avg Gripper Force: {session_stats['avg_force']:.1f}N")
        print(f"   Avg Stability: {session_stats['avg_stability']:.2f}")
        print(f"   Avg Lift Height: {session_stats['avg_lift_height']:.3f}m")
        
        # Category performance
        if session_stats['category_breakdown']:
            print(f"\\n📈 CATEGORY PERFORMANCE:")
            for category, rate in sorted(session_stats['category_breakdown'].items(), 
                                       key=lambda x: x[1], reverse=True):
                emoji = "🟢" if rate >= 0.6 else "🟡" if rate >= 0.3 else "🔴"
                print(f"   {emoji} {category}: {rate:.1%}")
        
        # Recent trend
        window_stats = self.get_rolling_window_stats()
        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}
        print(f"\\n{trend_emoji.get(window_stats.trend_direction, '➡️')} RECENT TREND:")
        print(f"   Last {self.window_size//60} minutes: "
              f"{window_stats.success_rate:.1%} success rate ({window_stats.trend_direction})")
        
        # Recommendations
        print(f"\\n💡 RECOMMENDATIONS:")
        if session_stats['success_rate'] >= 0.5:
            print("   ✅ Performance meets Phase 4.3 requirements")
            if window_stats.trend_direction == "improving":
                print("   📈 Positive trend - continue current approach")
            elif window_stats.trend_direction == "declining":
                print("   ⚠️ Recent decline - monitor for issues")
        else:
            print("   🔥 URGENT: Below 50% threshold - activate data generation")
            worst_category = min(session_stats['category_breakdown'].items(), 
                               key=lambda x: x[1])[0] if session_stats['category_breakdown'] else "Unknown"
            print(f"   🎯 Focus improvement on: {worst_category}")
        
        print("="*50)
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure."""
        return {
            'session_duration_minutes': 0.0,
            'total_attempts': 0,
            'total_successes': 0,
            'success_rate': 0.0,
            'avg_execution_time': 0.0,
            'avg_force': 0.0,
            'avg_stability': 0.0,
            'avg_lift_height': 0.0,
            'category_breakdown': {},
            'performance_rating': 'no_data',
            'attempts_per_minute': 0.0
        }
    
    def _rate_performance(self, success_rate: float) -> str:
        """Rate performance based on success rate."""
        if success_rate >= self.success_rate_thresholds['excellent']:
            return 'excellent'
        elif success_rate >= self.success_rate_thresholds['good']:
            return 'good'
        elif success_rate >= self.success_rate_thresholds['acceptable']:
            return 'acceptable'
        elif success_rate >= self.success_rate_thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def _detect_trend_direction(self) -> str:
        """Detect trend direction for recent performance."""
        if len(self.metrics_history) < 10:
            return "stable"
        
        # Compare first and second half of recent window
        recent_half_size = min(10, len(self.metrics_history) // 4)
        
        if len(self.metrics_history) < recent_half_size * 2:
            return "stable"
        
        first_half = self.metrics_history[-recent_half_size*2:-recent_half_size]
        second_half = self.metrics_history[-recent_half_size:]
        
        first_success_rate = sum(1 for m in first_half if m.success) / len(first_half)
        second_success_rate = sum(1 for m in second_half if m.success) / len(second_half)
        
        diff = second_success_rate - first_success_rate
        
        if diff > self.trend_sensitivity:
            return "improving"
        elif diff < -self.trend_sensitivity:
            return "declining"
        else:
            return "stable"
    
    def _determine_overall_trend(self, values: List[float]) -> str:
        """Determine overall trend from sequence of values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear regression to detect trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
        
        if slope > self.trend_sensitivity:
            return "improving"
        elif slope < -self.trend_sensitivity:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, success_rates: List[float]) -> float:
        """Calculate improvement rate per hour."""
        if len(success_rates) < 2 or not self.metrics_history:
            return 0.0
        
        # Time span of the data
        time_span_hours = (self.metrics_history[-1].timestamp - 
                          self.metrics_history[0].timestamp) / 3600
        
        if time_span_hours < 0.1:  # Less than 6 minutes
            return 0.0
        
        # Rate of change
        initial_rate = success_rates[0] if success_rates else 0.0
        final_rate = success_rates[-1] if success_rates else 0.0
        
        improvement_rate = ((final_rate - initial_rate) / time_span_hours) * 100
        
        return improvement_rate
    
    def save_performance_log(self, filename: Optional[str] = None) -> Path:
        """Save complete performance log to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_log_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Prepare data for serialization
        log_data = {
            'session_start': self.session_start_time,
            'session_end': time.time(),
            'total_metrics': len(self.metrics_history),
            'session_stats': self.get_current_session_stats(),
            'window_stats': asdict(self.get_rolling_window_stats()),
            'trends': asdict(self.analyze_performance_trends()),
            'detailed_metrics': [asdict(m) for m in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Performance log saved to: {filepath}")
        return filepath


# Global tracker instance for easy access
_global_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Get or create global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker

def reset_performance_tracker():
    """Reset global performance tracker."""
    global _global_tracker
    _global_tracker = None