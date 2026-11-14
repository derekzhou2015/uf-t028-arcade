"""Tests for PymunkPhysicsEngine's handling of sprite scale and velocity limits."""

import arcade
import pytest


def _square_sprite(scale=1.0) -> arcade.SpriteSolidColor:
    """A 32 x 32 sprite whose unscaled hit box is 32 x 32, centered on (0, 0)."""
    sprite = arcade.SpriteSolidColor(32, 32, color=arcade.color.WHITE)
    if scale != 1.0:
        sprite.scale = scale
    return sprite


def _shape_bounds(
    physics_engine: arcade.PymunkPhysicsEngine, sprite: arcade.Sprite
) -> tuple[float, float, float, float]:
    """Get the (left, right, bottom, top) bounds of a sprite's pymunk shape.

    The vertices are relative to the body, which sits unrotated at the origin
    in these tests, so they're also the shape's bounds around the sprite's
    center. The engine isn't stepped, so nothing moves them.
    """
    physics_object = physics_engine.get_physics_object(sprite)
    assert physics_object.body is not None
    assert tuple(physics_object.body.position) == (0.0, 0.0)
    assert physics_object.body.angle == 0.0
    shape = physics_object.shape
    assert shape is not None

    xs, ys = zip(*shape.get_vertices())
    return min(xs), max(xs), min(ys), max(ys)


def test_add_sprite_uniform_scale_shape():
    """A uniformly scaled sprite keeps a square shape."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite(scale=2.0)

    engine.add_sprite(sprite)

    assert _shape_bounds(engine, sprite) == pytest.approx((-32, 32, -32, 32), abs=1e-6)


def test_add_sprite_non_uniform_scale_shape():
    """Each axis of the shape uses the sprite's scale for that axis."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite(scale=(2.0, 0.5))

    engine.add_sprite(sprite)

    left, right, bottom, top = _shape_bounds(engine, sprite)
    assert (left, right, bottom, top) == pytest.approx((-32, 32, -8, 8), abs=1e-6)
    assert (right - left) == pytest.approx(64, abs=1e-6)
    assert (top - bottom) == pytest.approx(16, abs=1e-6)


def test_update_sprite_after_non_uniform_scale_change():
    """Refreshing the shape picks up a new non-uniform scale.

    The engine is unregistered from the sprite first, so that only the
    explicit call to ``update_sprite()`` can change the shape here.
    """
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite()

    engine.add_sprite(sprite)
    assert _shape_bounds(engine, sprite) == pytest.approx((-16, 16, -16, 16), abs=1e-6)

    sprite.physics_engines.remove(engine)
    sprite.scale = (3.0, 0.25)

    # Without the explicit refresh nothing would have changed the shape.
    assert _shape_bounds(engine, sprite) == pytest.approx((-16, 16, -16, 16), abs=1e-6)

    engine.update_sprite(sprite)

    assert _shape_bounds(engine, sprite) == pytest.approx((-48, 48, -4, 4), abs=1e-6)


def test_scale_change_keeps_custom_moment_of_inertia():
    """A moment passed to add_sprite() isn't overwritten by a resize."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite()

    engine.add_sprite(sprite, moment_of_inertia=arcade.PymunkPhysicsEngine.MOMENT_INF)

    sprite.scale = (2.0, 0.5)

    body = engine.get_physics_object(sprite).body
    assert body.moment == arcade.PymunkPhysicsEngine.MOMENT_INF
    assert _shape_bounds(engine, sprite) == pytest.approx((-32, 32, -8, 8), abs=1e-6)


def test_scale_change_updates_static_body_shape():
    """Static terrain also picks up a new scale."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite()

    engine.add_sprite(sprite, body_type=arcade.PymunkPhysicsEngine.STATIC)

    sprite.scale = (4.0, 0.5)

    assert _shape_bounds(engine, sprite) == pytest.approx((-64, 64, -8, 8), abs=1e-6)


def test_scale_change_refreshes_existing_shape():
    """Changing the sprite's scale updates the shape without a manual refresh."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite()

    engine.add_sprite(sprite)
    old_shape = engine.get_physics_object(sprite).shape

    sprite.scale_x = 2.0

    new_shape = engine.get_physics_object(sprite).shape
    assert new_shape is not old_shape
    assert _shape_bounds(engine, sprite) == pytest.approx((-32, 32, -16, 16), abs=1e-6)

    sprite.scale_y = 0.5
    assert _shape_bounds(engine, sprite) == pytest.approx((-32, 32, -8, 8), abs=1e-6)


def test_uniform_scale_backwards_compatible_shape_refresh():
    """Uniform scaling still produces the same shape as before, at any time."""
    engine = arcade.PymunkPhysicsEngine(gravity=(0, 0))
    sprite = _square_sprite(scale=1.5)

    engine.add_sprite(sprite)
    assert _shape_bounds(engine, sprite) == pytest.approx((-24, 24, -24, 24), abs=1e-6)

    sprite.scale = 0.5
    assert _shape_bounds(engine, sprite) == pytest.approx((-8, 8, -8, 8), abs=1e-6)


@pytest.mark.parametrize(
    "axis,gravity",
    (
        ("x", (-100, 0)),
        ("y", (0, -100)),
    ),
)
def test_velocity_limit_passed_at_add_time(axis, gravity):
    """A velocity limit passed to add_sprite() is enforced."""
    engine = arcade.PymunkPhysicsEngine(gravity=gravity)
    sprite = _square_sprite()

    kwargs = {f"max_{'horizontal' if axis == 'x' else 'vertical'}_velocity": 30}
    engine.add_sprite(sprite, **kwargs)

    engine.step(1.0)

    body = engine.get_physics_object(sprite).body
    assert getattr(body.velocity, axis) == pytest.approx(-30, abs=1e-6)


@pytest.mark.parametrize(
    "axis,gravity",
    (
        ("x", (-100, 0)),
        ("y", (0, -100)),
    ),
)
def test_velocity_limit_changed_at_runtime(axis, gravity):
    """A velocity limit changed at runtime applies, like the other axis's does."""
    engine = arcade.PymunkPhysicsEngine(gravity=gravity)
    sprite = _square_sprite()

    engine.add_sprite(sprite)

    attribute = f"max_{'horizontal' if axis == 'x' else 'vertical'}_velocity"
    setattr(sprite.pymunk, attribute, 40)

    engine.step(1.0)
    body = engine.get_physics_object(sprite).body
    assert getattr(body.velocity, axis) == pytest.approx(-40, abs=1e-6)

    # A second runtime change must be picked up as well.
    setattr(sprite.pymunk, attribute, 15)

    engine.step(1.0)
    body = engine.get_physics_object(sprite).body
    assert getattr(body.velocity, axis) == pytest.approx(-15, abs=1e-6)


def test_horizontal_and_vertical_limits_apply_together():
    """Both limits clamp their own axis while the other keeps its value."""
    engine = arcade.PymunkPhysicsEngine(gravity=(-100, -100))
    sprite = _square_sprite()

    engine.add_sprite(sprite, max_horizontal_velocity=20, max_vertical_velocity=5)

    engine.step(1.0)

    body = engine.get_physics_object(sprite).body
    assert body.velocity.x == pytest.approx(-20, abs=1e-6)
    assert body.velocity.y == pytest.approx(-5, abs=1e-6)
