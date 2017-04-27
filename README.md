# APILeekwars
some stuff to use the Leekwars API in python, very basic

```Python
import random
import APILeekwars as API

api = API.APILeekwars()
farmer_name = ####
password = ####
r = api.farmer.login_token(farmer_name, password)
if r["success"]:
    token = r["token"]
    r = api.garden.get_farmer_opponents(token)
    if r["success"]:
        random.seed()
        opponent = random.choice(r["opponents"])["id"]
        r = api.garden.start_farmer_fight(opponent, token)
        if r["success"]:
            print("https://leekwars.com/fight/{}".format(r["fight"]))
    r = api.farmer.disconnect(token)
```
