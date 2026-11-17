"""Task registry: name -> constructor. Tasks are imported lazily so the
corridor tests do not need MuJoCo."""


def make_env(task: dict):
    name = task["name"]
    kw = {k: v for k, v in task.items() if k not in ("name",)}
    if name == "pendulum":
        from .pendulum import Pendulum
        return Pendulum(**kw)
    if name == "cartpole":
        from .cartpole import CartPole
        return CartPole(**kw)
    if name == "go1":
        from .go1 import Go1Gait
        return Go1Gait(**kw)
    raise ValueError(f"unknown task {name}")
