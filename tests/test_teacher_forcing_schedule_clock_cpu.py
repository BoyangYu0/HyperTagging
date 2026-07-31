from hypertagging.training.scheduled_sampling import TeacherForcingSchedule


def test_thousand_step_schedule_is_not_at_endpoint_on_optimizer_step_ten():
    schedule = TeacherForcingSchedule(duration_steps=1000, start_probability=1.0, end_probability=0.2)
    assert schedule.probability(10) == 0.992
    assert schedule.probability(10) > schedule.end_probability

