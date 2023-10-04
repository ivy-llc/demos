import json
string = \
"""
{
  "type": "service_account",
  "project_id": "gpu-insatnce",
  "private_key_id": "a693769da2d4afbee63d3c37476d776d69edcd69",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCn6GqJg8ONfS1h\ntgTSw1R/tE/2Hlr2HwOpXtL5Au+KYCHwVMRw6a48liAKrmsNt5ny3u90ilg97czv\n8ZBfrl6p+0d40YGzJ+IXJ/IBdjIKioQdgi8zegnhhxZb7GsGdcF9yovzfSSNI2a/\nJAO7fOAHvFQQvCBhUmPAp0li0IS5ot/eCjyqZPxLLwZHRpgWqGhyfZqroyYi0XWq\nWybPegUJ7dfXxHyaqQEO1+LZCV2im3Oe8lUlCj6nNYl6YttyGI33PEuaLDdCL/cC\ndnz0umNMC/6OvGQGHbRfnr/WaZEXE3H+ffCvaduf7hZN0nqFUuBZ9y4JeRIf0rKL\nfmfmJIUlAgMBAAECggEACaDMKyKBHiXlakrzr9o/iCsgwq4u7RoOLbIgSuMeNUMZ\n7xwNP6RGt3asU7B4twqmK0UZWgds0/BE5iVl7/ahuwGLsaPh2hIZZtaFjAvpBq22\nsbJ7XmpLEsGaSJ8f6/jopXvt1oKAjA9RnvhDtoaMmiT0dRk0iiAdVDjDfoUSd0qF\n2FztO0t/IwLxMfwx51+Qbsr83uzuCp9/LCQdbz6q2VV6M3gSt+56m3dFCbWjNyxe\nAqWYbD2W5rk0ytbfSQPh/M4P13PMLV3THZ5fL5I2thkkHnftL0Qckxi64dee88OZ\nfpNQZdjqJCCE3duonspeJhu0/KoGiq56f3srkKZwKQKBgQDah5LHEhkQSZJlLPtF\nvuQDEwzxe2f0B0Fmyh9SGr8Jo/93lhQmwvRxPpVAuGv4TjStc2sJ2jvR2ESpcG/v\nZ/i1hL0oEkzxfVqps91tUMxLZMRQa2WURMmnqy9jFsfFyXmd61Z9U3+MtEPbZhHy\noR3dtMGwmCvJzhuwEQjIj9B8awKBgQDEssleayaP3yMWNU/TQgEsZifLQUZwIOhb\neA59sMW9IprkEzYc5JEaGoVbn4wHrPzVOVDcQ/5N/wb6vURi/LFrrcJDLBG1Itct\nMRFx4A3RlLob+esRNOj5Rqu4IOAAwI1YN+JyDcJA8+5Io94QGoAm/V4m5zgVLhYG\n10Np8TxorwKBgDpz7STmiGG87DW39HZRezf/ToUBV2sEzc+q3PnfQZ5Yiar5L+a2\nWyA3d0Dak7bVZG3I46a9Ayc8TBFui+Jj0zjqipY/XulxeGDQRbhJ1nuMNp60O3WP\neMkqlwBQRZt4Ktm862OWhBWk2knDYaWexxhYkz6jVYMjaHCN/eV34nKFAoGBAJ0+\n8IxyzJknoT0X0YDpu1ZrNFThEhaBuD+ncq44q1vQVlMOqKvgbb+YhBQDMEbSnC05\nbNMp6BsiTcoVlJ4ttQiIGEzq5h0QhUSzfWaz31g0tbO3z5wp2DWvsYe5gFoZlOV5\nh6X2Kd0TJOMdXWEyTlDdLG+dWNBVz3YYkOQY3loxAoGAQ5vlcmGuSbzf1UFg7R+n\nT1ZwqITQHLf8dKpPL6XKbujFHYpWciebkeDDLfCHdxfEAgbtqqk7RwCcoBPNCgaA\nZtULGqnmKRu5FLbH9GCxNjiwNlmkoYzKgnZjpoX04Hs7hv3y3tZKGY5PwhgUKf/z\nHeDwSyaJ4cgzn66OQe/aq/U=\n-----END PRIVATE KEY-----\n",
  "client_email": "demos-testing-ivy@gpu-insatnce.iam.gserviceaccount.com",
  "client_id": "107342639027122981894",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/demos-testing-ivy%40gpu-insatnce.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""
string = string.strip(' \n\t')
credentials = json.loads(string)