from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_script(name: str) -> Any:
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_legacy_visualization_env_entrypoint_is_removed():
    assert not (_SCRIPTS_DIR / "visualization_env.py").exists()


def test_visualize_task_env_keeps_canonical_defaults():
    mod = _load_script("visualize_task_env")

    args = mod._parse_args([])

    assert args.task == "Go2JoystickFlat"
    assert args.backend == "mujoco"
    assert args.num_envs == 4


def test_visualize_task_env_parses_explicit_args():
    mod = _load_script("visualize_task_env")

    args = mod._parse_args(
        [
            "--task",
            "Go2JoystickRough",
            "--backend",
            "motrix",
            "--num_envs",
            "8",
        ]
    )

    assert args.task == "Go2JoystickRough"
    assert args.backend == "motrix"
    assert args.num_envs == 8


def test_motrix_camera_kwargs_focuses_single_terrain_spawn():
    mod = _load_script("visualize_task_env")

    class FakeSpawn:
        def origins_for(self, env_ids):
            assert env_ids.tolist() == [0]
            return np.asarray([[10.0, 20.0, 0.25]], dtype=np.float64)

    class FakeEnv:
        _spawn = FakeSpawn()

    camera_kwargs = mod._motrix_camera_kwargs(FakeEnv(), 1)

    assert camera_kwargs == {
        "cam_lookat": [10.0, 20.0, 0.75],
        "cam_distance": 4.0,
        "cam_elevation": -25.0,
        "cam_azimuth": 135.0,
    }


def test_motrix_camera_kwargs_frames_multiple_terrain_spawns():
    mod = _load_script("visualize_task_env")

    class FakeSpawn:
        def origins_for(self, env_ids):
            assert env_ids.tolist() == [0, 1, 2, 3]
            return np.asarray(
                [
                    [-36.0, 36.0, 0.0],
                    [36.0, -12.0, 0.0],
                    [-4.0, -44.0, 0.0],
                    [-12.0, -44.0, 0.0],
                ],
                dtype=np.float64,
            )

    class FakeEnv:
        _spawn = FakeSpawn()

    camera_kwargs = mod._motrix_camera_kwargs(FakeEnv(), 4)

    assert camera_kwargs["cam_lookat"] == [0.0, -4.0, 0.5]
    assert camera_kwargs["cam_distance"] > 4.0
    assert camera_kwargs["cam_elevation"] == -25.0
    assert camera_kwargs["cam_azimuth"] == 135.0


def test_mujoco_visual_xml_paths_prefer_backend_visual_scene(tmp_path: Path):
    mod = _load_script("visualize_task_env")
    robot_xml = tmp_path / "robot.xml"
    visual_xml = tmp_path / "scene.xml"

    class FakeScene:
        model_file = str(robot_xml)

    class FakeBackend:
        scene_visual_model_file = str(visual_xml)

    class FakeEnv:
        _backend = FakeBackend()
        cfg = type("Cfg", (), {"scene": FakeScene()})()

    parent, robot = mod._mujoco_visual_xml_paths(FakeEnv())

    assert parent == visual_xml
    assert robot == robot_xml


def _keyboard_env(
    with_commands: bool = True,
    *,
    with_height_command: bool = False,
    env_cls_name: str = "Env",
    cfg_cls_name: str = "Cfg",
    module: str = "tests.fake_env",
    obs_contains_command: bool = False,
) -> Any:
    info: dict[str, Any] = (
        {"commands": np.asarray([[0.37, -0.23, 0.19]], dtype=np.float32)}
        if with_commands
        else {"steps": 0}
    )
    if obs_contains_command:
        info["commands"] = np.asarray([[0.37, -0.23, 0.19]], dtype=np.float32)
    if with_height_command:
        info["height_commands"] = np.asarray([[0.702]], dtype=np.float32)
    commands_cfg = (
        type(
            "Cmds",
            (),
            {
                "vel_limit": [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]],
                "heading_command": True,
                "resampling_time": 10.0,
                "height_range": [0.65, 0.754],
            },
        )()
        if with_commands
        else None
    )
    cfg_type = type(cfg_cls_name, (), {"__module__": module})
    cfg = cfg_type()
    cfg.commands = commands_cfg
    obs = (
        {"obs": np.asarray([[1.0, 0.37, -0.23, 0.19, 2.0]], dtype=np.float32)}
        if obs_contains_command
        else {"obs": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)}
    )
    state = type("State", (), {"info": info, "obs": obs})()
    env_type = type(env_cls_name, (), {"__module__": module})
    env = env_type()
    env.state = state
    env.cfg = cfg
    return env


def test_build_keyboard_commander_disabled_when_flag_off():
    mod = _load_script("play_interactive")
    args = type("Args", (), {"keyboard": False})()
    assert mod._build_keyboard_commander(_keyboard_env(), args) is None


def test_build_keyboard_commander_ignored_without_commands(capsys):
    mod = _load_script("play_interactive")
    args = type("Args", (), {"keyboard": True})()
    assert mod._build_keyboard_commander(_keyboard_env(with_commands=False), args) is None
    assert "no velocity 'commands'" in capsys.readouterr().out


def test_build_keyboard_commander_makes_keyboard_authoritative():
    mod = _load_script("play_interactive")
    env = _keyboard_env()
    args = type(
        "Args", (), {"keyboard": True, "keyboard_step_lin": 0.15, "keyboard_step_ang": 0.25}
    )()

    commander = mod._build_keyboard_commander(env, args)

    assert commander is not None
    assert commander.step_lin == 0.15
    assert commander.step_ang == 0.25
    # Heading P-control and resampling are turned off so they cannot fight the keyboard.
    assert env.cfg.commands.heading_command is False
    assert env.cfg.commands.resampling_time == 0.0
    assert env.state.info["commands"].tolist() == [[0.0, 0.0, 0.0]]


def test_velocity_arrows_require_velocity_command_task_and_policy_obs():
    mod = _load_script("play_interactive")

    joystick_env = _keyboard_env(
        env_cls_name="Go2WalkTask",
        cfg_cls_name="Go2JoystickCfg",
        module="unilab.envs.locomotion.go2.joystick",
        obs_contains_command=True,
    )
    manip_loco_env = _keyboard_env(
        env_cls_name="Go2ArmManipLocoEnv",
        cfg_cls_name="Go2ArmManipLocoCfg",
        module="unilab.envs.locomotion.go2_arm.manip_loco",
        obs_contains_command=True,
    )
    missing_obs_command_env = _keyboard_env(
        env_cls_name="Go2WalkTask",
        cfg_cls_name="Go2JoystickCfg",
        module="unilab.envs.locomotion.go2.joystick",
        obs_contains_command=False,
    )

    assert mod._should_render_velocity_arrows(joystick_env) is True
    assert mod._should_render_velocity_arrows(manip_loco_env) is False
    assert mod._should_render_velocity_arrows(missing_obs_command_env) is False


def test_handle_command_key_maps_drive_style_keys():
    mod = _load_script("play_interactive")
    commander = mod.KeyboardCommander.from_vel_limit([[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]])

    mod._handle_command_key(commander, mod._KEY_UP)  # forward (vx +)
    mod._handle_command_key(commander, mod._KEY_LEFT)  # turn left (vyaw +)
    assert commander.command == pytest.approx([0.1, 0.0, 0.2])

    mod._handle_command_key(commander, mod._KEY_RIGHT)  # turn right cancels yaw
    assert commander.command[2] == pytest.approx(0.0)

    before = commander.command.copy()
    mod._handle_command_key(commander, ord("A"))  # sideways left
    assert commander.command[1] == pytest.approx(0.1)
    mod._handle_command_key(commander, ord("d"))  # sideways right cancels vy
    assert commander.command.tolist() == before.tolist()
    mod._handle_command_key(commander, ord("q"))  # unmapped key is a no-op
    assert commander.command.tolist() == before.tolist()

    mod._handle_command_key(commander, mod._KEY_ENTER)  # full stop
    assert commander.command.tolist() == [0.0, 0.0, 0.0]


def test_height_keyboard_updates_external_target_with_configured_step():
    mod = _load_script("play_interactive")
    env = _keyboard_env(with_height_command=True)
    args = type("Args", (), {"keyboard": True, "keyboard_step_height": 0.01})()

    commander = mod._build_height_commander(env, args)

    assert commander is not None
    assert commander.target == pytest.approx(0.702)
    assert mod._handle_height_key(commander, ord("]")) is True
    assert commander.target == pytest.approx(0.712)
    assert mod._handle_height_key(commander, ord("[")) is True
    assert commander.target == pytest.approx(0.702)
    assert mod._handle_height_key(commander, ord("q")) is False
    assert env.cfg.commands.resampling_time == 0.0


def test_g1_standing_contract_flags_old_walking_only_run_config():
    mod = _load_script("play_interactive")
    run_config = {
        "config": {
            "env": {},
            "reward": {
                "scales": {
                    "feet_phase": 5.0,
                    "alive": 10.0,
                }
            },
        }
    }

    issues = mod._g1_standing_contract_issues(run_config)

    assert any("rel_standing_envs" in issue for issue in issues)
    assert any("reward.mode.enabled" in issue for issue in issues)
    assert any("feet_phase=5.0" in issue for issue in issues)


def test_g1_standing_contract_accepts_two_mode_run_config():
    mod = _load_script("play_interactive")
    run_config = {
        "config": {
            "env": {"commands": {"rel_standing_envs": 0.4}},
            "reward": {
                "scales": {"feet_phase": 0.0},
                "mode": {
                    "enabled": True,
                    "stand_terms": [
                        "stand_still",
                        "stand_action_l2",
                        "stand_dof_vel_l2",
                        "stand_lin_vel_xy_l2",
                        "stand_yaw_vel_l2",
                        "alive",
                    ],
                },
                "gait_constraint": {"freeze_phase_in_stand_mode": True},
            },
        }
    }

    assert mod._g1_standing_contract_issues(run_config) == []


def test_g1_sac_playback_warns_for_checkpoint_without_standing_contract(tmp_path):
    mod = _load_script("play_interactive")
    checkpoint = tmp_path / "model_1000.pt"
    checkpoint.write_bytes(b"")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "env": {},
                    "reward": {"scales": {"feet_phase": 5.0}},
                }
            }
        ),
        encoding="utf-8",
    )
    messages: list[str] = []

    issues = mod._warn_if_g1_sac_checkpoint_lacks_standing_contract(
        algo="sac",
        task_name="g1_walk_flat",
        checkpoint_path=str(checkpoint),
        log=messages.append,
    )

    assert issues
    assert any("standing/walking reward-mode contract" in message for message in messages)


def test_play_interactive_cli_accepts_distill_config_route():
    mod = _load_script("play_interactive")

    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "distill",
            "--task",
            "g1_walk_height",
            "--sim",
            "mujoco",
            "interactive.action_mode=policy",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)
    args = mod._build_play_args(cfg, algo=parsed.algo)

    assert parsed.algo == "distill"
    assert parsed.overrides[0] == "task=g1_walk_height/mujoco"
    assert cfg.algo.algo_log_name == "distill"
    assert cfg.training.task_name == "G1WalkHeight"
    assert args.algo == "distill"
    assert args.action_mode == "policy"
    assert args.policy_obs_mode == "auto"


def test_play_interactive_routes_distill_to_generic_session(monkeypatch):
    mod = _load_script("play_interactive")
    calls: dict[str, Any] = {}

    class FakeSession:
        env = object()

    def fake_create_distill_playback_session(**kwargs):
        calls.update(kwargs)
        return FakeSession(), "actor", "/tmp/model.pt"

    monkeypatch.setattr(
        mod, "create_distill_playback_session", fake_create_distill_playback_session
    )
    monkeypatch.setattr(mod, "_uses_native_mujoco_viewer_launch", lambda: True)
    monkeypatch.setattr(mod, "_can_launch_glfw_viewer", lambda: False)

    args = type(
        "Args",
        (),
        {
            "task": "G1WalkHeight",
            "load_run": "demo",
            "checkpoint": None,
            "action_mode": "policy",
            "policy_obs_mode": "actor",
            "algo_log_name": "distill",
            "log_root": None,
            "speed": 1.0,
            "start_paused": False,
            "algo": "distill",
        },
    )()
    cfg = mod.OmegaConf.create({"training": {"device": "cpu"}})

    mod.play_interactive(args, cfg=cfg, algo="distill")

    assert calls["playback_cfg"].algo_log_name == "distill"
    assert calls["playback_cfg"].action_mode == "policy"
    assert calls["cfg"] is cfg
    assert calls["device"] == "cpu"


def test_play_interactive_viewer_model_uses_shared_render_playback_resolver(
    tmp_path: Path, monkeypatch
):
    mod = _load_script("play_interactive")
    visual_xml = tmp_path / "scene.xml"
    visual_xml.write_text("<mujoco/>", encoding="utf-8")

    import mujoco

    loaded_binary: list[str] = []
    resolved: dict[str, object] = {}
    viewer_model = object()

    def fake_from_binary_path(path: str):
        loaded_binary.append(path)
        return viewer_model

    def fake_resolve_render_play_model_files(env, *, num_envs: int, tmp_dir: str | Path):
        resolved["env"] = env
        resolved["num_envs"] = num_envs
        resolved["tmp_dir"] = tmp_dir
        output_path = Path(tmp_dir) / "model_0.mjb"
        output_path.write_bytes(b"fake-mjb")
        return str(output_path)

    monkeypatch.setattr(mujoco.MjModel, "from_binary_path", fake_from_binary_path)
    monkeypatch.setattr(
        mod,
        "resolve_render_play_model_files",
        fake_resolve_render_play_model_files,
    )

    class FakeBackend:
        scene_visual_model_file = str(visual_xml)

    class FakeEnv:
        _backend = FakeBackend()

    env = FakeEnv()
    model = mod._load_viewer_model(env, use_env_visual_model=False)

    assert model is viewer_model
    assert len(loaded_binary) == 1
    assert Path(loaded_binary[0]).name == "model_0.mjb"
    assert resolved["env"] is env
    assert resolved["num_envs"] == 1
    assert Path(resolved["tmp_dir"]).name.startswith("unilab-interactive-viewer-")


def test_push_control_is_opt_in_and_requires_keyboard():
    mod = _load_script("play_interactive")
    for args in (
        type("Args", (), {"keyboard": True, "push_force_enabled": False})(),
        type("Args", (), {"keyboard": False, "push_force_enabled": True})(),
    ):
        assert mod._build_body_force_commander(object(), args) is None


def test_recovery_launcher_config_enables_push_with_explicit_force_units():
    mod = _load_script("play_interactive")
    cfg = mod._compose_interactive_config(
        "distill",
        [
            "task=g1_recovery_combined/mujoco",
            "interactive.keyboard=true",
            "interactive.push_force_enabled=true",
            "interactive.push_force_world=[0,175,0]",
        ],
    )
    args = mod._build_play_args(cfg, algo="distill")
    assert args.push_force_enabled
    assert args.push_force_world == (0.0, 175.0, 0.0)
    assert args.push_body_name == "torso_link"


def test_native_viewer_keys_are_handled_on_simulation_thread(monkeypatch):
    from threading import Thread, get_ident
    from types import SimpleNamespace

    mod = _load_script("play_interactive")
    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "distill",
            "--task",
            "g1_recovery_combined",
            "--sim",
            "mujoco",
            "interactive.keyboard=true",
            "interactive.push_force_enabled=true",
            "interactive.start_paused=true",
            "interactive.require_keyboard_command_obs=false",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)
    args = mod._build_play_args(cfg, algo="distill")
    owner_thread = get_ident()
    mutations = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(ctrl_dt=0.02),
        state=SimpleNamespace(
            info={
                "commands": np.zeros((1, 3)),
                "recovery_active": np.array([True]),
                "recovery_effective_commands": np.zeros((1, 3)),
            }
        ),
        set_autoreset=lambda value: None,
    )

    class Session:
        def reset(self):
            pass

        def reset_recovery_pose(self, *, supine):
            assert supine
            mutations.append(("reset", get_ident()))

        def set_external_command(self, command):
            env.state.info["commands"][:] = command

        def advance(self, controls):
            assert controls.consume_step_permission()
            mutations.append(("step", get_ident()))
            return True

        def physics_state(self):
            return np.zeros((1, 1))

    session = Session()
    session.env = env
    force = mod.BodyForceCommander(
        body_name="torso_link", body_ids=np.array([7]), force=np.array([[[150.0, 0.0, 0.0]]])
    )
    commander = mod.KeyboardCommander.from_vel_limit([[-1, -1, -1], [1, 1, 1]])
    monkeypatch.setattr(
        mod, "create_distill_playback_session", lambda **kw: (session, "actor", "/tmp/model.pt")
    )
    monkeypatch.setattr(mod, "_uses_native_mujoco_viewer_launch", lambda: False)
    monkeypatch.setattr(
        mod, "prepare_motion_overlay_selection", lambda *a, **kw: SimpleNamespace(enabled=False)
    )
    monkeypatch.setattr(mod, "_load_viewer_model", lambda *a, **kw: object())
    monkeypatch.setattr(mod, "_should_render_velocity_arrows", lambda *a, **kw: False)
    monkeypatch.setattr(mod, "_build_keyboard_commander", lambda *a: commander)
    monkeypatch.setattr(mod, "_build_height_commander", lambda *a: None)
    monkeypatch.setattr(mod, "_build_body_force_commander", lambda *a: force)
    monkeypatch.setattr(mod, "_resolve_focus_body_id", lambda *a: 0)
    monkeypatch.setattr(mod.mujoco, "MjData", lambda model: SimpleNamespace(xpos=np.zeros((1, 3))))
    monkeypatch.setattr(mod.mujoco, "mj_setState", lambda *a: None)
    monkeypatch.setattr(mod.mujoco, "mj_forward", lambda *a: None)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)

    class Viewer:
        user_scn = SimpleNamespace(ngeom=0)
        emitted = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def is_running(self):
            if self.emitted:
                return False
            self.emitted = True
            thread = Thread(target=lambda: [self.key_callback(ord(key)) for key in "FRCN"])
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert mutations == [], "viewer callbacks must not mutate physics or reset the episode"
            return True

        def sync(self):
            pass

    def launch(model, data, *, key_callback):
        viewer = Viewer()
        viewer.key_callback = key_callback
        return viewer

    monkeypatch.setattr(mod.mujoco.viewer, "launch_passive", launch)
    mod.play_interactive(args, cfg=cfg, algo="distill")
    assert mutations == [("reset", owner_thread), ("step", owner_thread)]
    assert force.force_for_step() is None


@pytest.mark.parametrize("enabled", [False, True])
def test_play_interactive_standing_assist_config_reaches_formal_factory(enabled):
    mod = _load_script("play_interactive")
    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "distill",
            "--task",
            "g1_recovery_combined",
            "--sim",
            "mujoco",
            "interactive.action_mode=policy",
            f"interactive.standing_assist={str(enabled).lower()}",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)
    args = mod._build_play_args(cfg, algo=parsed.algo)
    assert cfg.interactive.standing_assist is enabled
    assert cfg.training.task_name == "G1RecoveryCombined"
    assert args.action_mode == "policy"
