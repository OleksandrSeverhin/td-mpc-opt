import os
import warnings
import hydra
import torch
from omegaconf import DictConfig
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger


# Set environment variables to enable ManiSkill3 and accelerate rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'

# Suppress warnings
warnings.filterwarnings('ignore')

# Set PyTorch flags for performance and debugging
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


@hydra.main(config_name='generic', config_path='.')
def train(cfg: DictConfig):
    """
    Train a multi-task TD-MPC2 agent by distilling knowledge from a teacher.
    The script iterates through all specified tasks, training a student model
    on each while using a pre-trained teacher model for distillation.

    Args:
        cfg (DictConfig): The configuration object from Hydra.
    """
    assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
    device = torch.device('cuda')

    # Load and initialize teacher model (pre-trained on all tasks)
    cfg_teacher = parse_cfg(cfg.teacher_config)
    set_seed(cfg_teacher.seed)
    teacher_model = TDMPC2(cfg_teacher)
    try:
        teacher_model.load(cfg_teacher.checkpoint)
        print(colored("Teacher model loaded successfully.", 'green'))
    except FileNotFoundError:
        print(colored(f"Error: Teacher checkpoint not found at {cfg_teacher.checkpoint}", 'red'))
        return
    teacher_model.eval()

    # Load and initialize student model with the teacher
    cfg_student = parse_cfg(cfg.student_config)
    set_seed(cfg_student.seed)
    student_model = TDMPC2(cfg_student, teacher_model=teacher_model)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg_student.work_dir)
    logger = Logger(cfg_student)

    # Check for multitask configuration
    if not cfg_student.multitask:
        print(colored("Warning: 'multitask' is not set to True in student config. This script is designed for multi-task training.", 'yellow'))
    
    # --- Main Multi-Task Training Loop ---
    print('\nStarting multi-task training...')
    for task_id, task_name in enumerate(cfg_student.tasks):
        print(colored(f"\n--- Training on task: {task_name} (ID: {task_id}) ---", 'cyan', attrs=['bold']))

        # Initialize the environment and buffer for the current task
        env_cfg = cfg_student.copy()
        env_cfg.task_name = task_name
        env = make_env(env_cfg)
        buffer = Buffer(env_cfg)
        
        episode_count = 0
        obs, info = env.reset(seed=env_cfg.seed)
        episode_reward = 0
        t0 = True # Flag for the first step of an episode

        # Loop through a fixed number of steps for each task
        for step in tqdm(range(env_cfg.steps)):
            # Get action from the agent's policy. The `act` function handles
            # planning and exploration.
            with torch.no_grad():
                action = student_model.act(obs, t0=t0, eval_mode=False, task=task_id)

            # Step the environment with the selected action
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated

            # Add the transition to the replay buffer
            buffer.add(
                obs.cpu().numpy(),
                action.cpu().numpy(),
                reward,
                done,
                task_id
            )

            # Update the agent's models and policy
            if step >= env_cfg.train_after_steps:
                if step % env_cfg.train_every == 0:
                    student_model.update(buffer, step)

            # Update environment state and episode metrics
            obs = next_obs
            episode_reward += reward
            t0 = False

            if done:
                # Log episode metrics for the current task
                logger.log_metrics({
                    f'{task_name}/episode_reward': episode_reward,
                    f'{task_name}/episode_length': env.episode_length,
                }, step)

                # Reset environment for a new episode
                obs, info = env.reset()
                episode_count += 1
                episode_reward = 0
                t0 = True

                print(f"Task: {task_name} | Episode {episode_count} finished at step {step}. Reward: {episode_reward:.2f}")

    print('\nTraining completed successfully.')

if __name__ == '__main__':
    train()
