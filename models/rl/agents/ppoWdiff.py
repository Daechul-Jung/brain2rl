from models.rl.utils.diffusion import *
from models.rl.agents.ppo import *
from models.rl.utils.NeuralNetwork import *

class DiffusionPPOAgent(PPOAgent):
    """PPO Agent with diffusion-based policy"""
    
    def __init__(self, observation_dim, action_dim, device="cuda"):
        super().__init__(observation_dim, action_dim, device)
        
        # Replace policy network with diffusion policy
        self.policy_net = DiffusionPolicy(
            state_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_timesteps=1000
        ).to(self.device)
        
        # Keep value network as is
        self.value_net = NeuralNetwork(observation_dim, 1).to(self.device)
        
        # Update optimizer for new policy network
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def get_action(self, observation, training=True):
        """Get action using diffusion policy"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, mean, log_std = self.policy_net.get_action(obs_tensor, training)
        
        action_info = {
            'log_prob': log_prob.item(),
            'mean': mean.squeeze().cpu().numpy(),
            'std': log_std.squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info