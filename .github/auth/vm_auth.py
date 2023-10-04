import sys
import json
import paramiko

from google.auth import compute_engine
from googleapiclient import discovery
from google.oauth2.service_account import Credentials

def authenticate_vm(path):
    credentials = Credentials.from_service_file(path)
    return discovery.build('compute', 'v1', credentials=credentials)
def start_runner(creds, pkey, id = "gpu-insatnce", zone='us-central1-a', instance='demos-tests'):
    compute = authenticate_vm(creds)
    compute.instances().start(project=id, zone=zone, instance=instance).execute()
    request = compute.instances().get(project=id, zone=zone, instance=instance)
    response = request.execute()

    # Extract the external IP address of the instance
    external_ip = response['networkInterfaces'][0]['accessConfigs'][0]['natIP']

    # Establish an SSH connection to the instance
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(external_ip, pkey = key)

    # Execute the command on the instance
    stdin, stdout, stderr = ssh.exec_command('cd actions-runner; nohup ./run.sh')

    # Read the output of the command
    output = stdout.read().decode()

    # Close the SSH connection
    ssh.close()

    return output

if __name__ == "__main__":
    key = sys.argv[1]
    # Start the instance
    start_runner('/gcp_auth.json', pkey=key)



