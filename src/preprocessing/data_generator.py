"""
Synthetic Session Data Generator
Creates realistic web session logs with normal and attack patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from faker import Faker
import ipaddress

fake = Faker()
np.random.seed(42)
random.seed(42)

class SessionDataGenerator:
    def __init__(self):
        self.action_types = [
            'login', 'view_page', 'click_button', 'submit_form',
            'download_file', 'upload_file', 'search', 'logout',
            'view_profile', 'edit_profile', 'change_password',
            'view_dashboard', 'api_call', 'checkout', 'payment'
        ]
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
            'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)',
        ]
        
        self.locations = [
            {'city': 'New York', 'country': 'USA', 'lat': 40.7128, 'lon': -74.0060},
            {'city': 'London', 'country': 'UK', 'lat': 51.5074, 'lon': -0.1278},
            {'city': 'Tokyo', 'country': 'Japan', 'lat': 35.6762, 'lon': 139.6503},
            {'city': 'Mumbai', 'country': 'India', 'lat': 19.0760, 'lon': 72.8777},
            {'city': 'Sydney', 'country': 'Australia', 'lat': -33.8688, 'lon': 151.2093},
            {'city': 'Berlin', 'country': 'Germany', 'lat': 52.5200, 'lon': 13.4050},
            {'city': 'Toronto', 'country': 'Canada', 'lat': 43.6532, 'lon': -79.3832},
            {'city': 'Singapore', 'country': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
        ]
    
    def generate_session_id(self):
        """Generate a realistic session ID"""
        return fake.sha256()[:32]
    
    def generate_ip(self, base_ip=None):
        """Generate a random IP address"""
        if base_ip:
            return base_ip
        return fake.ipv4()
    
    def generate_device_fingerprint(self, consistent=True):
        """Generate device fingerprint"""
        if consistent:
            return fake.sha256()[:16]
        return fake.sha256()[:16]
    
    def get_random_location(self):
        """Get a random location"""
        return random.choice(self.locations)
    
    def calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(loc1['lat']), radians(loc1['lon'])
        lat2, lon2 = radians(loc2['lat']), radians(loc2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return 6371 * c  # Earth radius in km
    
    def generate_normal_session(self, session_id, user_id, start_time):
        """Generate a normal user session"""
        events = []
        
        # Consistent attributes for the session
        ip_address = self.generate_ip()
        location = self.get_random_location()
        user_agent = random.choice(self.user_agents)
        device_fingerprint = self.generate_device_fingerprint()
        
        # Session duration and actions
        duration = max(60, np.random.normal(3600, 1800))
        num_actions = max(5, int(np.random.normal(15, 10)))
        
        current_time = start_time
        
        for i in range(num_actions):
            action = random.choice(self.action_types)
            
            event = {
                'timestamp': current_time.isoformat(),
                'session_id': session_id,
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'device_fingerprint': device_fingerprint,
                'action': action,
                'city': location['city'],
                'country': location['country'],
                'latitude': location['lat'],
                'longitude': location['lon'],
                'is_attack': 0,
                'attack_type': 'none'
            }
            
            events.append(event)
            
            # Random time between actions
            time_delta = np.random.exponential(duration / num_actions)
            current_time += timedelta(seconds=time_delta)
        
        return events
    
    def generate_session_hijack(self, session_id, user_id, start_time):
        """Generate a session hijacking attack"""
        events = []
        
        # Legitimate part of session
        legit_ip = self.generate_ip()
        legit_location = self.get_random_location()
        legit_user_agent = random.choice(self.user_agents)
        legit_device_fp = self.generate_device_fingerprint()
        
        current_time = start_time
        num_legit_actions = random.randint(3, 8)
        
        # Legitimate actions
        for i in range(num_legit_actions):
            event = {
                'timestamp': current_time.isoformat(),
                'session_id': session_id,
                'user_id': user_id,
                'ip_address': legit_ip,
                'user_agent': legit_user_agent,
                'device_fingerprint': legit_device_fp,
                'action': random.choice(self.action_types[:8]),  # Normal actions
                'city': legit_location['city'],
                'country': legit_location['country'],
                'latitude': legit_location['lat'],
                'longitude': legit_location['lon'],
                'is_attack': 0,
                'attack_type': 'none'
            }
            events.append(event)
            current_time += timedelta(seconds=np.random.exponential(300))
        
        # HIJACK HAPPENS HERE - Different IP/Location/Device
        attacker_ip = self.generate_ip()
        attacker_location = random.choice([loc for loc in self.locations 
                                          if loc != legit_location])
        attacker_user_agent = random.choice([ua for ua in self.user_agents 
                                            if ua != legit_user_agent])
        attacker_device_fp = self.generate_device_fingerprint(consistent=False)
        
        # Very short time gap (hijacked)
        current_time += timedelta(seconds=random.randint(5, 60))
        
        num_attack_actions = random.randint(3, 10)
        
        for i in range(num_attack_actions):
            event = {
                'timestamp': current_time.isoformat(),
                'session_id': session_id,  # SAME session ID!
                'user_id': user_id,
                'ip_address': attacker_ip,  # Different IP
                'user_agent': attacker_user_agent,  # Different UA
                'device_fingerprint': attacker_device_fp,  # Different device
                'action': random.choice(['view_profile', 'edit_profile', 
                                        'change_password', 'download_file', 
                                        'api_call', 'payment']),
                'city': attacker_location['city'],
                'country': attacker_location['country'],
                'latitude': attacker_location['lat'],
                'longitude': attacker_location['lon'],
                'is_attack': 1,
                'attack_type': 'session_hijacking'
            }
            events.append(event)
            current_time += timedelta(seconds=np.random.exponential(200))
        
        return events
    
    def generate_session_fixation(self, session_id, user_id, start_time):
        """Generate a session fixation attack"""
        events = []
        
        # Attacker sets the session (pre-login)
        attacker_ip = self.generate_ip()
        attacker_location = self.get_random_location()
        attacker_user_agent = random.choice(self.user_agents)
        attacker_device_fp = self.generate_device_fingerprint()
        
        current_time = start_time
        
        # Attacker creates/accesses session before victim
        event = {
            'timestamp': current_time.isoformat(),
            'session_id': session_id,
            'user_id': None,  # Not logged in yet
            'ip_address': attacker_ip,
            'user_agent': attacker_user_agent,
            'device_fingerprint': attacker_device_fp,
            'action': 'access_page',
            'city': attacker_location['city'],
            'country': attacker_location['country'],
            'latitude': attacker_location['lat'],
            'longitude': attacker_location['lon'],
            'is_attack': 1,
            'attack_type': 'session_fixation'
        }
        events.append(event)
        
        # Time gap - attacker sends link to victim
        current_time += timedelta(minutes=random.randint(10, 120))
        
        # Victim logs in with the fixed session
        victim_ip = self.generate_ip()
        victim_location = random.choice([loc for loc in self.locations 
                                        if loc != attacker_location])
        victim_user_agent = random.choice(self.user_agents)
        victim_device_fp = self.generate_device_fingerprint(consistent=False)
        
        # Victim login event
        event = {
            'timestamp': current_time.isoformat(),
            'session_id': session_id,  # SAME session ID set by attacker
            'user_id': user_id,  # Now logged in
            'ip_address': victim_ip,
            'user_agent': victim_user_agent,
            'device_fingerprint': victim_device_fp,
            'action': 'login',
            'city': victim_location['city'],
            'country': victim_location['country'],
            'latitude': victim_location['lat'],
            'longitude': victim_location['lon'],
            'is_attack': 0,  # Victim doesn't know
            'attack_type': 'none'
        }
        events.append(event)
        
        # Victim's normal actions
        num_victim_actions = random.randint(3, 8)
        for i in range(num_victim_actions):
            current_time += timedelta(seconds=np.random.exponential(300))
            event = {
                'timestamp': current_time.isoformat(),
                'session_id': session_id,
                'user_id': user_id,
                'ip_address': victim_ip,
                'user_agent': victim_user_agent,
                'device_fingerprint': victim_device_fp,
                'action': random.choice(self.action_types),
                'city': victim_location['city'],
                'country': victim_location['country'],
                'latitude': victim_location['lat'],
                'longitude': victim_location['lon'],
                'is_attack': 0,
                'attack_type': 'none'
            }
            events.append(event)
        
        # Attacker uses the now-authenticated session
        current_time += timedelta(seconds=random.randint(10, 300))
        
        num_attack_actions = random.randint(3, 10)
        for i in range(num_attack_actions):
            event = {
                'timestamp': current_time.isoformat(),
                'session_id': session_id,  # SAME session ID
                'user_id': user_id,  # Now has access
                'ip_address': attacker_ip,  # Back to attacker IP
                'user_agent': attacker_user_agent,
                'device_fingerprint': attacker_device_fp,
                'action': random.choice(['view_profile', 'edit_profile', 
                                        'change_password', 'download_file']),
                'city': attacker_location['city'],
                'country': attacker_location['country'],
                'latitude': attacker_location['lat'],
                'longitude': attacker_location['lon'],
                'is_attack': 1,
                'attack_type': 'session_fixation'
            }
            events.append(event)
            current_time += timedelta(seconds=np.random.exponential(200))
        
        return events
    
    def generate_dataset(self, num_normal=1000, num_hijack=100, num_fixation=100):
        """Generate complete dataset"""
        all_events = []
        
        start_date = datetime.now() - timedelta(days=30)
        
        print(f"Generating {num_normal} normal sessions...")
        for i in range(num_normal):
            session_id = self.generate_session_id()
            user_id = f"user_{random.randint(1000, 9999)}"
            start_time = start_date + timedelta(
                seconds=random.randint(0, 30*24*3600)
            )
            
            events = self.generate_normal_session(session_id, user_id, start_time)
            all_events.extend(events)
        
        print(f"Generating {num_hijack} session hijacking attacks...")
        for i in range(num_hijack):
            session_id = self.generate_session_id()
            user_id = f"user_{random.randint(1000, 9999)}"
            start_time = start_date + timedelta(
                seconds=random.randint(0, 30*24*3600)
            )
            
            events = self.generate_session_hijack(session_id, user_id, start_time)
            all_events.extend(events)
        
        print(f"Generating {num_fixation} session fixation attacks...")
        for i in range(num_fixation):
            session_id = self.generate_session_id()
            user_id = f"user_{random.randint(1000, 9999)}"
            start_time = start_date + timedelta(
                seconds=random.randint(0, 30*24*3600)
            )
            
            events = self.generate_session_fixation(session_id, user_id, start_time)
            all_events.extend(events)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nDataset generated:")
        print(f"Total events: {len(df)}")
        print(f"Total sessions: {df['session_id'].nunique()}")
        print(f"Attack events: {df['is_attack'].sum()}")
        print(f"Normal events: {(df['is_attack'] == 0).sum()}")
        
        return df

if __name__ == "__main__":
    generator = SessionDataGenerator()
    df = generator.generate_dataset(
        num_normal=1000,
        num_hijack=100,
        num_fixation=100
    )
    
    # Save dataset
    df.to_csv('data/raw/session_logs.csv', index=False)
    print("\nDataset saved to data/raw/session_logs.csv")
    
    # Print sample
    print("\nSample events:")
    print(df.head(10))
    
    print("\nAttack type distribution:")
    print(df['attack_type'].value_counts())
