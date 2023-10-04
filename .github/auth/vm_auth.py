import sys
import paramiko
from google.auth import compute_engine
from googleapiclient import discovery
from google.oauth2.service_account import Credentials

def authenticate_vm(creds):
    credentials = Credentials.from_service_account_info(creds)
    return discovery.build('compute', 'v1', credentials=credentials)
def start_runner(document, pkey, id = "gpu-insatnce", zone='us-central1-a', instance='demos-tests'):
    compute = authenticate_vm(document)
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
    creds, key = sys.argv[1], sys.argv[2]
    # Start the instance
    start_runner(eval(creds), key)



