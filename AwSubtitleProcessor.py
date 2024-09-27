import json


class AwSubtitleProcessor:

    def __init__(self, min_char_length, max_char_length):
        # Constructor initializes min and max character lengths
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length

    def cyrillic_to_latin(self, text):
        cyrillic_to_latin_map = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
            'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
            'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
            'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
        }

        return ''.join(cyrillic_to_latin_map.get(char, char) for char in text)
    def extract_subtitle_info(self, segments):
        words = []
        itterable = segments["segments"]
        for segment in itterable:
            for word_segment in segment["words"]:
                word = word_segment.get('word', None)
                start = word_segment.get('start', None)
                end = word_segment.get('end', None)

                if word is None or start is None or end is None:
                    print(f"Skipping Segment: {word_segment}")
                    continue

                words.append(word_segment)
        print(words[-1])
        return words

    def prepare_params_for_write_vtt(self, segments):
        subtitle_lines = []
        current_line = ""
        line_start = 0
        max_char_length = self.max_char_length  # Assuming this is set elsewhere in your class

        for i, word_data in enumerate(segments):
            try:
                # Check if adding the new word exceeds the max_char_length
                if len(current_line) + len(word_data['word']) + 1 <= max_char_length:  # +1 for the space
                    if current_line == "":
                        # Set start time for the line
                        line_start = word_data['start']
                    current_line += word_data['word'] + " "  # Add word and a space to the current line
                else:
                    # End the current subtitle line
                    line_end = segments[i - 1]['end'] if i > 0 else word_data['end']

                    subtitle_lines.append({
                        "line": self.cyrillic_to_latin(current_line.strip()),  # Strip trailing space
                        "start": line_start,
                        "end": line_end
                    })

                    # Start a new line
                    current_line = word_data['word'] + " "  # Start a new line with the current word
                    line_start = word_data['start']  # New start time

            except Exception as e:
                print(f"An error occurred while processing segments: {e}")
                print(f"word data: {word_data}")  # Log the entire word_data that caused the error

        # Append the last line if it's not empty
        if current_line:
            line_end = segments[-1]['end']
            subtitle_lines.append({
                "line": current_line.strip(),
                "start": line_start,
                "end": line_end
            })

        return subtitle_lines

    def format_timestamp(self, seconds: float, is_vtt: bool = False):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        separator = '.' if is_vtt else ','
        hours_marker = f"{hours:02d}:"
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"
        )


