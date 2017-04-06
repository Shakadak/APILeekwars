#!/usr/bin/env python3

import os
import requests
import json

class APIFarmer:
    """docstring for wat"""
    def __init__(self, session):
        self.session = session
        self.url = "https://leekwars.com/api/farmer"

    def activate(self, farmer_id, code):
        url = self.url + "/activate/"
        return self.session.post(url, data = {"farmer_id" : farmer_id, "code" : code}).json()

    def change_country(self, country_code, token):
        url = self.url + "/change-country/"
        return self.session.post(url, data = {"country_code" : country_code, "token" : token}).json()

    def change_password(self, password, new_password, token):
        url = self.url + "/change-password/"
        return self.session.post(url, data = {"password" : password, "new_password" : new_password, "token" : token}).json()

    def disconnect(self, token):
        url = self.url + "/disconnect/"
        return self.session.post(url, data = {"token" : token}).json()

    def get(self, farmer_id):
        url = self.url + "/get/" + farmer_id
        return self.session.get(url).json()

    def get_connected(self):
        url = self.url + "/get-connected"
        return self.session.get(url).json()

    def get_from_token(self, token):
        url = self.url + "/get-from-token/" + token
        return self.session.get(url).json()

    def login_token(self, login, password):
        url = self.url + "/login-token/"
        return self.session.post(url, data = {"login" : login,
                                            "password" : password}).json()

    def register(self, login, password, email, leek_name, godfather):
        url = self.url + "/register/"
        return self.session.post(url, data = {"login" : login,
                                            "password" : password,
                                            "email" : email,
                                            "leek_name" : leek_name,
                                            "godfather" : godfather}).json()

    def regiter_tournament(self, token):
        url = self.url + "/register-tournament/"
        return self.session.post(url, data = {"token" : token}).json()

    def set_avatar(self, avatar, token):
        url = self.url + "/set-avatar/"
        return self.session.post(url, data = {"token" : token}, files = {"avatar" : avatar}).json()

    def set_github(self, github, token):
        url = self.url + "/set-github/"
        return self.session.post(url, data = {"github" : github, "token" : token}).json()

    def set_in_garden(self, in_garden, token):
        url = self.url + "/set-in-garden/"
        return self.session.post(url, data = {"in_garden" : in_garden, "token" : token}).json()

    def set_wesbite(self, website, token):
        url = self.url + "/set-website/"
        return self.session.post(url, data = {"website" : website, "token" : token}).json()

    def unregister(self, password, delete_forum_messages, token):
        url = self.url + "/unregister/"
        return self.session.post(url, data = {"password" : password, "delete_forum_messages" : delete_forum_messages, "token" : token}).json()

    def unregister_tournament(self, token):
        url = self.url + "/unregister-tournament/"
        return self.session.post(url, data = {"token" : token}).json()

    def update(self, token):
        url = self.url + "/update/"
        return self.session.post(url, data = {"token" : token}).json()

class APIFight:
    """docstring for APIFight"""
    def __init__(self, session):
        self.session = session
        self.url = "https://leekwars.com/api/fight"

    def comment(self, fight_id, comment, token):
        url = self.url + "/comment/"
        return self.session.post(url, data = {"fight_id" : fight_id, "comment" : comment, "token" : token}).json()

    def get(self, fight_id):
        url = self.url + "/get/" + str(fight_id)
        return self.session.get(url).json()

    def get_logs(self, fight_id, token):
        url = self.url + "/get/" + str(fight_id) + "/" + token
        return self.session.get(url).json()

class APIFunction():
    """docstring for APIFunction"""
    def __init__(self, session):
        self.session = session
        self.url = "https://leekwars.com/api/function"

    def get_all(self):
        url = self.url + "/get-all/"
        return self.session.get(url).json()

    def get_all(self):
        url = self.url + "/get-categories/"
        return self.session.get(url).json()

class APIGarden():
    """docstring for APIGarden"""
    def __init__(self, session):
        self.session = session
        self.url = "https://leekwars.com/api/garden"

    def get(self, token):
        url = self.url + "/get-all/" + token
        return self.session.get(url).json()

    def get_composition_opponents(self, composition, token):
        url = self.url + "/get-composition-opponents/" + str(composition) + "/" + token
        return self.session.get(url).json()

    def get_farmer_challenge(self, target, token):
        url = self.url + "/get-farmer-challenge/" + str(target) + "/" + token
        return self.session.get(url).json()

    def get_farmer_opponents(self, token):
        url = self.url + "/get-farmer-opponents/" + token
        return self.session.get(url).json()

    def get_leek_opponents(self, leek_id, token):
        url = self.url + "/get-leek-opponents/" + str(leek_id) + "/" + token
        return self.session.get(url).json()

    def get_solo_challenge(self, leek_id, token):
        url = self.url + "/get-solo-challenge/" + str(leek_id) + "/" + token
        return self.session.get(url).json()

    def start_farmer_challenge(self, target_id, token):
        url = self.url + "/start-farmer-challenge/"
        return self.session.post(url, data = {"target_id" : target_id, "token" : token}).json()

    def start_farmer_fight(self, target_id, token):
        url = self.url + "/start-farmer-fight/"
        return self.session.post(url, data = {"target_id" : target_id, "token" : token}).json()

    def start_solo_challenge(self, leek_id, target_id, token):
        url = self.url + "/start-solo-challenge/"
        return self.session.post(url, data = {"leek_id" : leek_id, "target_id" : target_id, "token" : token}).json()

    def start_solo_fight(self, leek_id, target_id, token):
        url = self.url + "/start-solo-fight/"
        return self.session.post(url, data = {"leek_id" : leek_id, "target_id" : target_id, "token" : token}).json()

    def start_team_fight(self, composition_id, target_id, token):
        url = self.url + "/start-team-fight/"
        return self.session.post(url, data = {"composition_id" : composition_id, "target_id" : target_id, "token" : token}).json()

class APILeekwars():
    """docstring for APILeekwars"""
    def __init__(self):
        self.session = requests.Session()
        self.farmer = APIFarmer(self.session)
        self.fight = APIFight(self.session)
        self.function = APIFunction(self.session)
        self.garden = APIGarden(self.session)


def init_session():
    fn = "leekwars.data"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        pass
    leekwars = APILeekwars()
    farmer_name = "PumpkinAreBetter"
    token = leekwars.farmer.login_token(farmer_name, base_data["farmers"][farmer_name])["token"]
    print(token)
    r = leekwars.farmer.get_from_token(token)
    print(r)
    r = leekwars.fight.get(21049389)
    print(r)
    r = leekwars.farmer.disconnect(token)
    print(r)
    #print(json.dumps(r, sort_keys=True, indent=4))

if __name__ == '__main__':
    init_session()
