"""
Formatting Helper Functions
Utilities for formatting data for display
"""

from datetime import datetime, timedelta

class FormatHelpers:
    """Helper functions for formatting data"""
    
    @staticmethod
    def format_duration(milliseconds):
        """
        Convert milliseconds to human-readable duration
        
        Args:
            milliseconds: Duration in milliseconds
            
        Returns:
            Formatted string (e.g., "3:45", "1h 23m", "2d 5h")
        """
        if milliseconds is None:
            return "N/A"
        
        seconds = int(milliseconds / 1000)
        
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}:{secs:02d}"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"
    
    @staticmethod
    def format_number(num, precision=0):
        """
        Format number with thousand separators
        
        Args:
            num: Number to format
            precision: Decimal places
            
        Returns:
            Formatted string (e.g., "1,234.56")
        """
        if num is None:
            return "N/A"
        
        if precision == 0:
            return f"{int(num):,}"
        else:
            return f"{num:,.{precision}f}"
    
    @staticmethod
    def format_percentage(value, total, precision=1):
        """
        Calculate and format percentage
        
        Args:
            value: Part value
            total: Total value
            precision: Decimal places
            
        Returns:
            Formatted percentage string (e.g., "45.2%")
        """
        if total == 0:
            return "0%"
        
        percentage = (value / total) * 100
        return f"{percentage:.{precision}f}%"
    
    @staticmethod
    def format_date(date_string):
        """
        Format date string to readable format
        
        Args:
            date_string: ISO format date string
            
        Returns:
            Formatted date (e.g., "Jan 15, 2024")
        """
        if not date_string or date_string == "N/A":
            return "N/A"
        
        try:
            date_obj = datetime.fromisoformat(str(date_string).replace('Z', '+00:00'))
            return date_obj.strftime("%b %d, %Y")
        except:
            return str(date_string)
    
    @staticmethod
    def format_time_ago(date_string):
        """
        Format date as time ago (e.g., "2 months ago")
        
        Args:
            date_string: ISO format date string
            
        Returns:
            Human-readable time difference
        """
        if not date_string:
            return "N/A"
        
        try:
            date_obj = datetime.fromisoformat(str(date_string).replace('Z', '+00:00'))
            now = datetime.now(date_obj.tzinfo)
            diff = now - date_obj
            
            if diff.days == 0:
                return "Today"
            elif diff.days == 1:
                return "Yesterday"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif diff.days < 365:
                months = diff.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            else:
                years = diff.days // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
        except:
            return "N/A"
    
    @staticmethod
    def truncate_text(text, max_length=50, suffix="..."):
        """
        Truncate text to maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum character length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def format_popularity_label(popularity):
        """
        Get descriptive label for popularity score
        
        Args:
            popularity: Popularity score (0-100)
            
        Returns:
            Descriptive label
        """
        if popularity is None:
            return "Unknown"
        
        if popularity >= 80:
            return "🔥 Viral Hit"
        elif popularity >= 70:
            return "⭐ Very Popular"
        elif popularity >= 50:
            return "📈 Popular"
        elif popularity >= 30:
            return "🎵 Moderately Known"
        elif popularity >= 10:
            return "💎 Hidden Gem"
        else:
            return "🔍 Deep Cut"
    
    @staticmethod
    def format_mood_description(mood):
        """
        Get description for mood category
        
        Args:
            mood: Mood category
            
        Returns:
            Description string
        """
        descriptions = {
            'Happy': 'Uplifting and joyful vibes',
            'Sad': 'Melancholic and emotional',
            'Energetic': 'High-energy and exciting',
            'Chill': 'Relaxed and laid-back'
        }
        return descriptions.get(mood, 'Unknown mood')
    
    @staticmethod
    def format_audio_feature_description(feature, value):
        """
        Get description for audio feature value
        
        Args:
            feature: Feature name
            value: Feature value (0-1)
            
        Returns:
            Description string
        """
        descriptions = {
            'danceability': {
                'high': 'Perfect for dancing',
                'medium': 'Moderately danceable',
                'low': 'Not very danceable'
            },
            'energy': {
                'high': 'Intense and powerful',
                'medium': 'Moderate energy',
                'low': 'Calm and peaceful'
            },
            'valence': {
                'high': 'Positive and cheerful',
                'medium': 'Neutral mood',
                'low': 'Dark and somber'
            },
            'acousticness': {
                'high': 'Acoustic instruments',
                'medium': 'Mix of acoustic/electric',
                'low': 'Electronic/synthesized'
            }
        }
        
        if value >= 0.7:
            level = 'high'
        elif value >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return descriptions.get(feature.lower(), {}).get(level, 'Unknown')
    
    @staticmethod
    def get_emoji_rating(value, max_value=5):
        """
        Convert numeric rating to emoji stars
        
        Args:
            value: Rating value
            max_value: Maximum rating value
            
        Returns:
            Emoji star string
        """
        if value is None:
            return "⭐" * max_value
        
        stars = int((value / 100) * max_value) if value <= 100 else int(value)
        stars = min(stars, max_value)
        
        full_stars = "⭐" * stars
        empty_stars = "☆" * (max_value - stars)
        
        return full_stars + empty_stars
    
    @staticmethod
    def format_list_with_and(items):
        """
        Format list with proper grammar (e.g., "A, B, and C")
        
        Args:
            items: List of items
            
        Returns:
            Formatted string
        """
        if not items:
            return ""
        if len(items) == 1:
            return str(items[0])
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        
        return ", ".join(items[:-1]) + f", and {items[-1]}"
