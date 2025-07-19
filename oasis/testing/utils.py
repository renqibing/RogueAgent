import os
import yaml
import shutil
from pathlib import Path
from oasis.social_agent.bad_agents_generator import MaliciousAgentGenerator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class FileManager:
    def __init__(self, gen_num=10, num_timesteps=30, exp_name='attack', yaml_dir=None, model_path='', new_host=None, new_ports=None):
        self.root_dir = ROOT_DIR
        print(ROOT_DIR)
        # self.db_dir = self.root_dir / 'data/simu_db'
        # self.csv_dir = self.root_dir / 'csv'
        self.yaml_dir = os.path.join(
            self.root_dir,
            yaml_dir
        )
        self.new_db_dir = f'data/simu_db/yaml_{exp_name}'
        self.new_csv_dir = f'data/twitter_dataset/anonymous_topic_200_1h_{exp_name}'
        self.new_yaml_dir = f'scripts/twitter_simulation/align_with_real_world/yaml_200_{exp_name}/sub1'
        self.gen_num = gen_num
        self.num_timesteps = num_timesteps
        self.model_path = model_path 
        self.new_host = new_host
        self.new_ports = new_ports
        
    def create_dirs(self):
        """Create new directories for db, csv, and yaml."""
        for dir_path in [self.new_db_dir, self.new_csv_dir, self.new_yaml_dir]:
            dir_path = os.path.join(
                self.root_dir,
                dir_path
            )
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                print(f"{dir_path} already exists")

    def traverse_and_process_yaml(self):
        """Traverse the yaml dir, read each file, process data, and update the yaml file."""
        # Check if the yaml directory exists
        if not os.path.exists(self.yaml_dir):
            print(f"{self.yaml_dir} does not exist.")
            return

        # Process each YAML file in the yaml directory
        for yaml_file in Path(self.yaml_dir).glob('*.yaml'):    
            print(f"Processing file: {yaml_file}")
            with open(yaml_file, 'r') as file:
                yaml_data = yaml.safe_load(file)

            # Extract db_path and csv_path
            db_path = yaml_data['data']['db_path']
            csv_path = yaml_data['data']['csv_path']

            # Generate and save agents (assuming this function is available)
            agent_generator = MaliciousAgentGenerator(csv_path, self.new_csv_dir, self.root_dir)
            agent_generator.generate_agents(n=self.gen_num)

            # Update YAML data with the new paths
            yaml_data['data']['db_path'] = str(os.path.join(self.new_db_dir, db_path.split('/')[-1]))
            yaml_data['data']['csv_path'] = str(os.path.join(self.new_csv_dir, csv_path.split('/')[-1]))

            # Update number of agents in YAML data
            yaml_data['model']['num_agents'] += self.gen_num
            yaml_data['model']['cfgs'][0]['num'] += self.gen_num

            # Update number of simulation steps 
            yaml_data['simulation']['num_timesteps'] = self.num_timesteps

            # Update server url 
            yaml_data = self.update_server_url(yaml_data)

            # Save updated YAML file to the new YAML directory
            new_yaml_file = os.path.join(
                self.root_dir,
                self.new_yaml_dir,
                yaml_file.name
            )            
            with open(new_yaml_file, 'w') as file:
                yaml.dump(yaml_data, file, default_flow_style=False)
            print(f"Updated YAML saved to: {new_yaml_file}")

    def update_server_url(self, yaml_data):
        """
        Updates the host and ports in the `server_url` section of the YAML data.
        
        :param yaml_data: Dictionary representing the YAML data
        :param new_host: New host to set
        :param new_ports: New list of ports to set
        :return: Updated YAML data
        """
        # Check if 'server_url' exists in the data
        if 'inference' in yaml_data:
            if 'server_url' in yaml_data['inference']:
                for server in yaml_data['inference']['server_url']:
                    server['host'] = self.new_host  # Update host
                    server['ports'] = self.new_ports  # Update ports
        yaml_data['model']['cfgs'][0]['server_url'] = f'http://{self.new_host}:{self.new_ports[0]}/v1'
        yaml_data['model']['cfgs'][0]['model_path'] = self.model_path
        yaml_data['inference']['model_path'] = self.model_path

        return yaml_data

    def process(self):
        """Main processing function."""
        self.create_dirs()
        # generate bad agents 
        self.traverse_and_process_yaml()


if __name__ == "__main__":
    # Example usage
    # root_dir = "/path/to/your/root/directory"
    gen_num = 10
    num_timesteps = 1
    exp_name = f'attack_{gen_num}_{num_timesteps}'
    yaml_dir = 'scripts/twitter_simulation/align_with_real_world/yaml_200/sub1'
    model_path = "/mnt/hwfile/trustai/lijun/models/llama3-8b-instruct"
    new_host = '10.140.1.71'  # Replace with the new host
    new_ports = [8612]  # Replace with the new ports list
    file_manager = FileManager(gen_num, num_timesteps, exp_name, yaml_dir, model_path, new_host, new_ports)
    file_manager.process()

