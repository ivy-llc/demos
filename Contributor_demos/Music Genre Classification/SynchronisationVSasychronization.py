Synchronisation 

UNIFY_KEY=#ENTERUNIFYKEY
!pip install unifyai

from unify import Unify
unify = Unify(
    api_key=UNIFY_KEY,
    endpoint="llama-2-13b-chat@anyscale"
)
respole"
)
responsense = unify.generate(user_prompt="Hello Llama! Who was Isaac Newton?")
print(response)

Asynchronize

from unify import AsyncUnify
import asyncio
import nest_asyncio
nest_asyncio.apply()

async_unify = AsyncUnify(
   api_key=UNIFY_KEY,
   endpoint="llama-2-13b-chat@anyscale"
)

async def main():
   responses = await async_unify.generate(user_prompt="Hello Llama! Who was Isaac Newton?")
   print(responses)

asyncio.run(main())



