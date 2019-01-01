import java.util.ArrayList;
import java.util.List;

public class GFM {

    public static void convert(String srcFilePath, String detFilePath) {

        List<String> contents = TxtFileUtil.readLineByLine(srcFilePath);

        List<String> newContents = new ArrayList<String>();

        boolean isCode = false;

        for (String text : contents) {
            text = SpecialsChars.Escape(text, isCode);

            if (text.contains("```")) {
                isCode = !isCode;
            } else if (!isCode) {
                text = MathJaxHelper.changeMathJaxToCodeCogs(text);
                text = CenterTag.convert(text);
                text = AsteriskChars.addSpace(text);
            }

            newContents.add(text);
        }

        newContents = TOC.changeTOCToGeneratedCatalogue(newContents);

        for (String str : newContents) System.out.println(str);

        TxtFileUtil.writeLineByLine(newContents, detFilePath);
    }
}
