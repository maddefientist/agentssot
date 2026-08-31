from app.main import app


def test_passphrase_enrollment_and_generated_shell_installers_are_not_shipped():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/enroll/auto" not in paths
    assert "/enroll/portal" not in paths
    assert "/enroll/bootstrap.sh" not in paths
    assert "/enroll/install-plugin.sh" not in paths
    assert "/enroll" in paths
