public class SpecialsChars {

    public static String Escape(String text, boolean isCode) {

        if (!isCode) {
            text = text.replaceAll("([^\\\\])~", "$1\\\\~");
        }

        // System.out.println(" ".equals(" "));  // false, one is 160，one is 32
        text = text.replaceAll(" ", " ");

        return text;
    }
}
